# AI Chess Arena — Parallel Self-Play

A local-first AI-vs-AI chess research prototype with a React dashboard, FastAPI backend, PyTorch policy networks, `python-chess`, SQLite game history, live WebSocket updates, checkpoints, parallel self-play batches, and a frozen-baseline comparison endpoint.

**Important:** this is an experimental learning system, not a strong chess engine. More self-play games do not guarantee stronger chess. The current reward-learning method is deliberately simple; use repeated evaluation runs and larger samples before drawing conclusions.

## Requirements

- Python 3.11+
- Node.js 20+
- CPU is supported. A CUDA-enabled PyTorch build may be installed separately for GPU use.

## Windows setup

Open two terminals.

### 1. Backend

```powershell
cd ai-chess-arena\backend
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### 2. Frontend

```powershell
cd ai-chess-arena\frontend
npm install
npm run dev
```

Open http://localhost:5173. API docs: http://localhost:8000/docs.

## Linux/macOS setup

```bash
cd ai-chess-arena/backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
cd ai-chess-arena/frontend
npm install
npm run dev
```

## What's included

- **Parallel self-play:** training batches run 1–64 separate chess games concurrently; default is 16. Each game has its own board, move history, PGN and result.
- **Batch-stable policy:** all games in a batch use the same model weights. Model updates happen serially after the batch to avoid concurrent optimizer writes.
- **Real gradient updates:** after each completed game, the selected moves are replayed through the policy network and the optimizer updates model weights using the game's result.
- **Draw/repetition handling:** repeated positions receive a sampling penalty, and draws receive a small negative training reward. This can reduce repetitive play but does not mean all draws are bad.
- **Checkpoints:** model and optimizer states are saved periodically and on stop/end; use Save checkpoint for an immediate save.
- **Metrics/history:** SQLite stores games, PGNs, losses, outcomes and checkpoint metadata.
- **Baseline comparison:** Training includes a button that runs eight games against a frozen copy of the models captured when the backend starts. Current and baseline colors alternate. Stop training first. Restarting the backend resets this comparison baseline to the models loaded at startup.
- **Live arena:** shows one focused game's board while other games continue in parallel; the training panel reports the parallel-game setting and active game count.

## Training notes

The policy uses a compact fully connected network over 12 piece planes and scores from-square/to-square actions. Illegal moves are excluded. This is a baseline experiment, not AlphaZero: there is no MCTS, value network, opening book, or engine-grade evaluation. The old-vs-current comparison uses a small fixed number of games by default and is noisy; run it repeatedly and increase the game count through the API before judging progress.

## API

- `GET /api/health`
- `GET /api/dashboard`
- `GET /api/models`
- `GET /api/training/status`
- `POST /api/training/start` with `{ "target_episodes": 64, "max_moves": 160, "learning_rate": 0.001, "games_per_batch": 16 }`
- `POST /api/training/pause`
- `POST /api/training/resume`
- `POST /api/training/stop`
- `POST /api/training/checkpoint`
- `POST /api/evaluation/run?games=8` (even number from 2 to 32; stop training first)
- `GET /api/evaluation/latest`
- `GET /api/games?limit=50&offset=0`
- `GET /api/games/{id}`
- `GET /api/games/{id}/pgn`
- `GET /api/metrics`
- `WS /ws/live`

## Data paths

- SQLite: `data/arena.sqlite3`
- Latest checkpoints: `checkpoints/white_latest.pt` and `checkpoints/black_latest.pt`

## Troubleshooting

If you see `No module named 'numpy'`, activate the backend virtual environment and run `python -m pip install -r requirements.txt` again. If model evaluation returns a 409 error, stop training first so the comparison is consistent.

This is for local development. Do not expose the API publicly without authentication and stricter network configuration.
