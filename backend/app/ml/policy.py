from pathlib import Path
import torch
from torch import nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PolicyNet(nn.Module):
    """Small policy network. Input is a 12x8x8 piece-plane tensor; output is from/to logits."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(12 * 8 * 8, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 64 * 64)
        )
    def forward(self, x):
        return self.net(x)

def encode_board(board):
    x = torch.zeros((12, 8, 8), dtype=torch.float32)
    piece_index = {"p":0, "n":1, "b":2, "r":3, "q":4, "k":5}
    for square, piece in board.piece_map().items():
        plane = piece_index[piece.symbol().lower()] + (0 if piece.color else 6)
        row, col = 7 - square // 8, square % 8
        x[plane, row, col] = 1.0
    return x

class Agent:
    def __init__(self, color):
        self.color = color
        self.model = PolicyNet().to(DEVICE)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.steps = 0
        self.last_loss = None
        self.version = 1
        self.status = "Ready"
        self.path = Path(__file__).resolve().parents[3] / "checkpoints" / f"{color}_latest.pt"
        self.load()
    def select_move(self, board, explore=True):
        import chess
        legal = list(board.legal_moves)
        if not legal:
            return None, None
        state = encode_board(board).unsqueeze(0).to(DEVICE)
        logits = self.model(state)[0]
        indices = torch.tensor([m.from_square * 64 + m.to_square for m in legal], device=DEVICE)
        legal_logits = logits[indices].clone()
        # Discourage immediate repetitions while keeping them legal when they are best.
        # Apply the penalty only to candidate moves that recreate a previously seen position.
        for i, candidate in enumerate(legal):
            board.push(candidate)
            repeats = board.is_repetition(2)
            board.pop()
            if repeats:
                legal_logits[i] -= 1.5
        # Promotion variants share from/to logits; a tiny deterministic preference promotes to queen.
        probs = torch.softmax(legal_logits, dim=0)
        if not explore:
            idx = int(torch.argmax(probs).item())
        else:
            idx = int(torch.multinomial(probs, 1).item())
        move = legal[idx]
        if move.promotion:
            same = [m for m in legal if m.from_square == move.from_square and m.to_square == move.to_square]
            move = next((m for m in same if m.promotion == chess.QUEEN), same[0])
        log_prob = torch.log_softmax(legal_logits, dim=0)[idx]
        return move, log_prob
    def update_episode(self, log_probs, reward, gamma=0.99):
        if not log_probs:
            return None
        returns, running = [], 0.0
        for _ in reversed(log_probs):
            running = reward + gamma * running
            returns.append(running)
        returns.reverse()
        rt = torch.tensor(returns, dtype=torch.float32, device=DEVICE)
        # Keep the sign of the terminal reward; per-episode normalization can
        # accidentally turn losing trajectories into positive updates.
        rt = rt.clamp(-10.0, 10.0)
        loss = -(torch.stack(log_probs) * rt).mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.steps += 1
        self.last_loss = float(loss.detach().cpu())
        return self.last_loss
    def update_trajectory(self, trajectory, reward, gamma=0.99):
        """Recompute policy log-probabilities from a completed game, then update once.

        trajectory contains (FEN-before-move, UCI move) pairs. Recomputing after a
        parallel batch keeps all games in that batch on a frozen policy snapshot.
        """
        import chess
        if not trajectory:
            return None
        log_probs = []
        for fen, uci in trajectory:
            board = chess.Board(fen)
            try:
                move = chess.Move.from_uci(uci)
            except ValueError:
                continue
            if move not in board.legal_moves:
                continue
            logits = self.model(encode_board(board).unsqueeze(0).to(DEVICE))[0]
            legal = list(board.legal_moves)
            indices = torch.tensor([m.from_square * 64 + m.to_square for m in legal], device=DEVICE)
            selected = next((i for i, candidate in enumerate(legal) if candidate == move), None)
            if selected is None:
                continue
            log_probs.append(torch.log_softmax(logits[indices], dim=0)[selected])
        if not log_probs:
            return None
        returns, running = [], 0.0
        for _ in reversed(log_probs):
            running = reward + gamma * running
            returns.append(running)
        returns.reverse()
        rt = torch.tensor(returns, dtype=torch.float32, device=DEVICE).clamp(-10.0, 10.0)
        loss = -(torch.stack(log_probs) * rt).mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.steps += 1
        self.last_loss = float(loss.detach().cpu())
        self.version += 1
        return self.last_loss

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        torch.save({"state_dict": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                    "steps": self.steps, "version": self.version}, tmp)
        tmp.replace(self.path)
    def load(self):
        if self.path.exists():
            try:
                data = torch.load(self.path, map_location=DEVICE, weights_only=False)
                self.model.load_state_dict(data["state_dict"])
                if "optimizer" in data: self.optimizer.load_state_dict(data["optimizer"])
                self.steps = int(data.get("steps", 0)); self.version = int(data.get("version", 1))
            except Exception:
                self.status = "Error"
