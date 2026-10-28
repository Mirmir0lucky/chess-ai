import asyncio, copy, json, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import chess, chess.pgn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from .database import init_db, connection
from .ml.policy import Agent, encode_board, DEVICE

ROOT = Path(__file__).resolve().parents[1]
app = FastAPI(title="AI Chess Arena API", version="0.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
white_agent, black_agent = Agent("white"), Agent("black")
# Frozen reference opponents captured at startup; used for a repeatable old-vs-current check.
baseline_white_model = copy.deepcopy(white_agent.model).to(DEVICE).eval()
baseline_black_model = copy.deepcopy(black_agent.model).to(DEVICE).eval()
last_evaluation = None
clients: set[WebSocket] = set()
worker: threading.Thread | None = None
state_lock = threading.Lock()
loop: asyncio.AbstractEventLoop | None = None
wins = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}
worker_state = {"status": "Ready", "episodes": 0, "target_episodes": 100, "max_moves": 160, "games_per_batch": 16, "active_games": 0, "started_at": None, "last_error": None, "paused": False, "stop": False, "active_game": None}

def now(): return datetime.now(timezone.utc).isoformat()
def emit(event: dict[str, Any]):
    if loop and clients:
        asyncio.run_coroutine_threadsafe(broadcast(event), loop)
async def broadcast(event):
    dead = []
    payload = json.dumps(event)
    for ws in list(clients):
        try: await ws.send_text(payload)
        except Exception: dead.append(ws)
    for ws in dead: clients.discard(ws)

def model_payload(agent):
    return {"name": f"{agent.color.title()} Policy", "color": agent.color, "version": agent.version,
            "status": agent.status, "checkpoint_exists": agent.path.exists(), "checkpoint_path": str(agent.path),
            "parameters": sum(p.numel() for p in agent.model.parameters()), "steps": agent.steps,
            "last_loss": agent.last_loss, "learning_rate": agent.optimizer.param_groups[0]["lr"]}


def baseline_move(model, board):
    import torch
    legal = list(board.legal_moves)
    if not legal: return None
    with torch.no_grad():
        logits = model(encode_board(board).unsqueeze(0).to(DEVICE))[0]
        indices = torch.tensor([m.from_square * 64 + m.to_square for m in legal], device=DEVICE)
        idx = int(torch.argmax(logits[indices]).item())
    move = legal[idx]
    if move.promotion:
        same = [m for m in legal if m.from_square == move.from_square and m.to_square == move.to_square]
        move = next((m for m in same if m.promotion == chess.QUEEN), same[0])
    return move

def play_one_game():
    board = chess.Board(); game = chess.pgn.Game.from_board(board)
    game.headers["Event"] = "AI Chess Arena Parallel Self-Play"
    game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
    started = time.time(); started_iso = now(); move_no = 0
    with connection() as c:
        cur = c.execute("INSERT INTO games(started_at,start_fen,white_version,black_version) VALUES(?,?,?,?)",
                        (started_iso, board.fen(), white_agent.version, black_agent.version))
        game_id = cur.lastrowid
    white_trajectory, black_trajectory = [], []
    san_moves: list[str] = []
    emit({"type":"game_start", "game_id":game_id, "fen":board.fen(), "moves":[], "turn":"white", "status":"playing"})
    termination = "max_moves"
    while not board.is_game_over(claim_draw=True) and move_no < worker_state["max_moves"]:
        while worker_state["paused"] and not worker_state["stop"]: time.sleep(0.1)
        if worker_state["stop"]: termination = "stopped"; break
        mover = white_agent if board.turn == chess.WHITE else black_agent
        trajectory = white_trajectory if board.turn == chess.WHITE else black_trajectory
        fen_before = board.fen()
        move, _ = mover.select_move(board, explore=True)
        if move is None: break
        san = board.san(move); from_sq, to_sq = chess.square_name(move.from_square), chess.square_name(move.to_square)
        trajectory.append((fen_before, move.uci()))
        board.push(move); game.end().add_variation(move); move_no += 1; san_moves.append(san)
        emit({"type":"move", "game_id":game_id, "fen":board.fen(), "move":{"from":from_sq,"to":to_sq,"san":san,"uci":move.uci()},
              "san":san,"moves":san_moves.copy(), "turn":"white" if board.turn else "black", "move_number":move_no,
              "check":board.is_check(), "status":"playing"})
        # Tiny delay keeps live updates readable without serializing the parallel games.
        time.sleep(0.015)
    if termination == "stopped":
        with connection() as c: c.execute("DELETE FROM games WHERE id=?", (game_id,))
        return None
    if board.is_game_over(claim_draw=True):
        outcome = board.outcome(claim_draw=True); result = outcome.result() if outcome else "1/2-1/2"
        termination = outcome.termination.name.lower() if outcome else "draw"
    else:
        result = "1/2-1/2"; termination = "move_limit"
    game.headers["Result"] = result
    pgn = str(game)
    duration = time.time()-started
    with connection() as c:
        c.execute("UPDATE games SET ended_at=?,result=?,termination=?,moves_count=?,duration_seconds=?,pgn=? WHERE id=?",
                  (now(), result, termination, move_no, duration, pgn, game_id))
    emit({"type":"game_end", "game_id":game_id,"result":result,"termination":termination,"moves_count":move_no,"duration_seconds":round(duration,2),"fen":board.fen(),"pgn":pgn})
    # Draws get a small negative reward so repeated sterile draws are not entirely ignored.
    if result == "1-0": white_reward, black_reward = 1.0, -1.0
    elif result == "0-1": white_reward, black_reward = -1.0, 1.0
    else: white_reward, black_reward = -0.05, -0.05
    return {"game_id":game_id, "result":result, "white_trajectory":white_trajectory, "black_trajectory":black_trajectory,
            "white_reward":white_reward, "black_reward":black_reward, "moves_count":move_no, "duration":duration}

def finish_batch(completed):
    # Update models serially only after all games in the batch have finished, so all
    # parallel games play against the same policy snapshot and no optimizer races occur.
    for record in completed:
        if record is None: continue
        wloss = white_agent.update_trajectory(record["white_trajectory"], record["white_reward"])
        bloss = black_agent.update_trajectory(record["black_trajectory"], record["black_reward"])
        with state_lock:
            worker_state["episodes"] += 1
            wins[record["result"]] = wins.get(record["result"], 0) + 1
            episode = worker_state["episodes"]
            total = max(1, sum(wins.values()))
            active = worker_state["active_games"]
        with connection() as c:
            c.execute("INSERT INTO training_metrics(created_at,episode,white_loss,black_loss,white_win_rate,black_win_rate,draw_rate) VALUES(?,?,?,?,?,?,?)",
                      (now(), episode, wloss, bloss, wins["1-0"]/total, wins["0-1"]/total, wins["1/2-1/2"]/total))
        emit({"type":"training_metric", "episode":episode, "white_loss":wloss,"black_loss":bloss,"models":[model_payload(white_agent),model_payload(black_agent)],"active_games":active})
        if episode % 5 == 0:
            white_agent.save(); black_agent.save()
            with connection() as c:
                for a in (white_agent, black_agent): c.execute("INSERT INTO checkpoints(created_at,color,version,path,steps) VALUES(?,?,?,?,?)", (now(),a.color,a.version,str(a.path),a.steps))
            emit({"type":"checkpoint", "created_at":now()})

def worker_main():
    try:
        worker_state["status"] = "Training"; white_agent.status = black_agent.status = "Training"
        with connection() as c:
            cur = c.execute("INSERT INTO training_runs(started_at,status,episodes) VALUES(?,?,0)", (now(),"Training")); run_id=cur.lastrowid
        emit({"type":"training_status","status":"Training"})
        while worker_state["episodes"] < worker_state["target_episodes"] and not worker_state["stop"]:
            while worker_state["paused"] and not worker_state["stop"]: time.sleep(0.1)
            if worker_state["stop"]: break
            remaining = worker_state["target_episodes"] - worker_state["episodes"]
            batch_size = min(worker_state["games_per_batch"], remaining)
            worker_state["active_games"] = batch_size
            emit({"type":"parallel_batch_start", "active_games":batch_size, "games_per_batch":worker_state["games_per_batch"]})
            completed = []
            with ThreadPoolExecutor(max_workers=batch_size, thread_name_prefix="chess-selfplay") as pool:
                futures = [pool.submit(play_one_game) for _ in range(batch_size)]
                for future in as_completed(futures):
                    try:
                        completed.append(future.result())
                    except Exception as exc:
                        worker_state["last_error"] = str(exc)
                        emit({"type":"error", "message":str(exc)})
            worker_state["active_games"] = 0
            finish_batch(completed)
        worker_state["status"] = "Ready" if not worker_state["stop"] else "Stopped"
        white_agent.status = black_agent.status = worker_state["status"]
        white_agent.save(); black_agent.save()
        with connection() as c: c.execute("UPDATE training_runs SET ended_at=?,status=?,episodes=?,last_error=? WHERE id=?", (now(),worker_state["status"],worker_state["episodes"],worker_state["last_error"],run_id))
        emit({"type":"training_status","status":worker_state["status"]})
    except Exception as e:
        worker_state["status"]="Error"; worker_state["last_error"]=str(e)
        white_agent.status = black_agent.status = "Error"
        emit({"type":"error","message":str(e)})
    finally:
        worker_state["active_games"] = 0; worker_state["paused"] = False; worker_state["stop"] = False

class TrainingConfig(BaseModel):
    target_episodes: int = Field(default=100, ge=1, le=100000)
    max_moves: int = Field(default=160, ge=20, le=500)
    learning_rate: float = Field(default=0.001, gt=0, le=0.1)
    games_per_batch: int = Field(default=16, ge=1, le=64)

@app.on_event("startup")
def startup():
    global loop
    loop = asyncio.get_running_loop(); init_db()
    with connection() as c:
        rows = c.execute("SELECT result, COUNT(*) n FROM games WHERE result IS NOT NULL GROUP BY result").fetchall()
        total = c.execute("SELECT COUNT(*) n FROM games WHERE result IS NOT NULL").fetchone()["n"]
    for row in rows: wins[row["result"]] = row["n"]
    worker_state["episodes"] = total

@app.get("/api/health")
def health(): return {"status":"ok", "device":str(__import__('torch').device('cuda' if __import__('torch').cuda.is_available() else 'cpu'))}
@app.get("/api/dashboard")
def dashboard():
    with connection() as c:
        total = c.execute("SELECT COUNT(*) n FROM games WHERE result IS NOT NULL").fetchone()["n"]
        rows = c.execute("SELECT result,COUNT(*) n FROM games WHERE result IS NOT NULL GROUP BY result").fetchall()
        avg = c.execute("SELECT AVG(moves_count) moves,AVG(duration_seconds) duration FROM games WHERE result IS NOT NULL").fetchone()
        latest = c.execute("SELECT MAX(created_at) t FROM checkpoints").fetchone()["t"]
        duration = c.execute("SELECT SUM(duration_seconds) s FROM games WHERE result IS NOT NULL").fetchone()["s"] or 0
    counts={r["result"]:r["n"] for r in rows}; denom=max(total,1)
    return {"total_games":total,"episodes":worker_state["episodes"],"white_wins":counts.get("1-0",0),"black_wins":counts.get("0-1",0),"draws":counts.get("1/2-1/2",0),
            "white_win_rate":counts.get("1-0",0)/denom,"black_win_rate":counts.get("0-1",0)/denom,"draw_rate":counts.get("1/2-1/2",0)/denom,
            "avg_moves":avg["moves"] or 0,"avg_game_seconds":avg["duration"] or 0,"training_seconds":duration,"last_checkpoint":latest,"training_status":worker_state["status"],"active_games":worker_state["active_games"]}
@app.get("/api/models")
def models(): return [model_payload(white_agent), model_payload(black_agent)]
@app.get("/api/training/status")
def training_status(): return {**worker_state, "models":[model_payload(white_agent),model_payload(black_agent)]}
@app.post("/api/training/start")
def start_training(config: TrainingConfig = TrainingConfig()):
    global worker
    if worker and worker.is_alive(): raise HTTPException(409,"Training worker is already running")
    worker_state.update({"status":"Training","target_episodes":worker_state["episodes"]+config.target_episodes,"max_moves":config.max_moves,"games_per_batch":config.games_per_batch,"started_at":now(),"last_error":None,"paused":False,"stop":False})
    for agent in (white_agent,black_agent): agent.optimizer.param_groups[0]["lr"] = config.learning_rate
    worker=threading.Thread(target=worker_main,daemon=True); worker.start()
    return {"status":"started","target_episodes":worker_state["target_episodes"],"new_episodes":config.target_episodes,"games_per_batch":config.games_per_batch}
@app.post("/api/training/pause")
def pause_training():
    if not worker or not worker.is_alive(): raise HTTPException(409,"No active training worker")
    worker_state["paused"]=True; worker_state["status"]="Paused"; emit({"type":"training_status","status":"Paused"}); return {"status":"Paused"}
@app.post("/api/training/resume")
def resume_training():
    if not worker or not worker.is_alive(): raise HTTPException(409,"No active training worker")
    worker_state["paused"]=False; worker_state["status"]="Training"; emit({"type":"training_status","status":"Training"}); return {"status":"Training"}
@app.post("/api/training/stop")
def stop_training():
    if not worker or not worker.is_alive(): raise HTTPException(409,"No active training worker")
    worker_state["stop"]=True; worker_state["paused"]=False; return {"status":"stopping"}
@app.post("/api/evaluation/run")
def run_evaluation(games: int = 8):
    global last_evaluation
    if games < 2 or games > 32 or games % 2 != 0:
        raise HTTPException(400, "games must be an even number from 2 to 32")
    if worker and worker.is_alive():
        raise HTTPException(409, "Stop training before running a model comparison so the result is consistent")
    results = {"current_wins": 0, "baseline_wins": 0, "draws": 0, "games": games, "avg_moves": 0}
    move_counts = []
    for index in range(games):
        board = chess.Board(); current_is_white = index % 2 == 0; plies = 0
        while not board.is_game_over(claim_draw=True) and plies < 160:
            if board.turn == chess.WHITE:
                if current_is_white:
                    move, _ = white_agent.select_move(board, explore=False)
                else:
                    move = baseline_move(baseline_white_model, board)
            else:
                if current_is_white:
                    move = baseline_move(baseline_black_model, board)
                else:
                    move, _ = black_agent.select_move(board, explore=False)
            if move is None: break
            board.push(move); plies += 1
        outcome = board.outcome(claim_draw=True)
        result = outcome.result() if outcome else "1/2-1/2"
        move_counts.append(plies)
        if result == "1/2-1/2": results["draws"] += 1
        elif (result == "1-0") == current_is_white: results["current_wins"] += 1
        else: results["baseline_wins"] += 1
    results["avg_moves"] = round(sum(move_counts) / max(1, len(move_counts)), 2)
    results["current_score_rate"] = round((results["current_wins"] + 0.5 * results["draws"]) / games, 4)
    results["baseline_note"] = "Frozen model snapshot captured when backend started; restart backend to reset the baseline."
    last_evaluation = results
    return results

@app.get("/api/evaluation/latest")
def latest_evaluation():
    return last_evaluation

@app.post("/api/training/checkpoint")
def save_checkpoint():
    white_agent.save(); black_agent.save()
    with connection() as c:
        for a in (white_agent,black_agent): c.execute("INSERT INTO checkpoints(created_at,color,version,path,steps) VALUES(?,?,?,?,?)",(now(),a.color,a.version,str(a.path),a.steps))
    emit({"type":"checkpoint","created_at":now()}); return {"status":"saved"}
@app.get("/api/games")
def games(limit:int=50, offset:int=0, result:str|None=None):
    if limit<1 or limit>200: raise HTTPException(400,"limit must be between 1 and 200")
    with connection() as c:
        if result: rows=c.execute("SELECT id,started_at,ended_at,result,termination,moves_count,duration_seconds,white_version,black_version FROM games WHERE result=? ORDER BY id DESC LIMIT ? OFFSET ?",(result,limit,offset)).fetchall()
        else: rows=c.execute("SELECT id,started_at,ended_at,result,termination,moves_count,duration_seconds,white_version,black_version FROM games ORDER BY id DESC LIMIT ? OFFSET ?",(limit,offset)).fetchall()
    return [dict(r) for r in rows]
@app.get("/api/games/{game_id}")
def game_detail(game_id:int):
    with connection() as c: row=c.execute("SELECT * FROM games WHERE id=?",(game_id,)).fetchone()
    if not row: raise HTTPException(404,"Game not found")
    return dict(row)
@app.get("/api/games/{game_id}/pgn")
def game_pgn(game_id:int):
    with connection() as c: row=c.execute("SELECT pgn FROM games WHERE id=?",(game_id,)).fetchone()
    if not row or not row["pgn"]: raise HTTPException(404,"PGN not found")
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(row["pgn"],media_type="application/x-chess-pgn")
@app.get("/api/metrics")
def metrics(limit:int=200):
    with connection() as c: rows=c.execute("SELECT created_at,episode,white_loss,black_loss,white_win_rate,black_win_rate,draw_rate FROM training_metrics ORDER BY id DESC LIMIT ?",(limit,)).fetchall()
    return [dict(r) for r in reversed(rows)]
@app.websocket("/ws/live")
async def websocket_live(ws:WebSocket):
    await ws.accept(); clients.add(ws)
    await ws.send_text(json.dumps({"type":"snapshot","training":worker_state,"models":[model_payload(white_agent),model_payload(black_agent)]}))
    try:
        while True: await ws.receive_text()
    except WebSocketDisconnect: clients.discard(ws)
