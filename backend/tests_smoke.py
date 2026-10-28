"""Smoke tests: run from backend with `python tests_smoke.py`."""
import chess
from app.ml.policy import Agent

board = chess.Board()
agent = Agent("white")
move, _ = agent.select_move(board)
assert move in board.legal_moves, f"Illegal move selected: {move}"
fen = board.fen()
uci = move.uci()
board.push(move)
assert board.ply() == 1
loss = agent.update_trajectory([(fen, uci)], 1.0)
assert loss is not None
assert agent.steps >= 1
print("PASS: legal move, trajectory-based gradient update, and training step.")
