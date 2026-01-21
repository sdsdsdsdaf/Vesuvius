import torch
import torch.nn as nn

import torch
import torch.nn as nn

class TTAPredictor(nn.Module):
    def __init__(self, model: nn.Module, device: torch.device, use_rot90: bool = True):
        super().__init__()
        
        self.model = model
        self.device = device
        self.use_rot90 = use_rot90  # rot0 + rot90(k=1) if True

    @staticmethod
    def _flip_hw(x: torch.Tensor, flip_h: bool, flip_w: bool) -> torch.Tensor:
        # H = -2, W = -1 for (B, C, D, H, W)
        if flip_h:
            x = torch.flip(x, dims=[-2])
        if flip_w:
            x = torch.flip(x, dims=[-1])
        return x

    @staticmethod
    def _rot90_hw(x: torch.Tensor, k: int) -> torch.Tensor:
        # Rotate in the H/W plane only (no interpolation)
        # k in {0,1,2,3}
        return torch.rot90(x, k=k, dims=(-2, -1))

    @torch.no_grad()
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Perform TTA prediction by averaging predictions over:
          - flips: identity, flip_h, flip_w, flip_hw
          - optional rot90 in H/W plane: rot0 and rot90(k=1)

        Args:
            inputs (torch.Tensor): (B, C, D, H, W) e.g. (B, 1, D, H, W)
        Returns:
            torch.Tensor: Averaged output (Logit) tensor of shape (B, C_out, D, H, W)
        """
        self.model.eval()

        # If you want, you can uncomment this to ensure correct device,
        # but typically SWI already feeds patches on the right device.
        # inputs = inputs.to(self.device, non_blocking=True)

        # Define TTA transforms
        flip_cases = [(False, False), (True, False), (False, True), (True, True)]
        rot_cases = [0, 1] if self.use_rot90 else [0]  # rot0 + rot90

        acc = None
        n = 0

        for rk in rot_cases:
            x_rot = self._rot90_hw(inputs, rk)

            for fh, fw in flip_cases:
                x = self._flip_hw(x_rot, fh, fw)
                y = self.model(x)

                # Invert transforms on output (reverse order)
                y = self._flip_hw(y, fh, fw)
                y = self._rot90_hw(y, (4 - rk) % 4)

                acc = y.float() if acc is None else (acc + y.float())
                n += 1

        return acc / float(n)

if __name__ == "__main__":
    class IdentityModel(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x
        
    model = IdentityModel()
    tta_predictor = TTAPredictor(model, device=torch.device("cpu"), use_rot90=True)

    x = torch.randn(2, 1, 160, 160, 160)  # (B, C, D, H, W)
    y:torch.Tensor = tta_predictor(x)
    print(y.shape)  # Should be (2, 1, 8, 16, 16)
    print("All values same as Input: ", torch.allclose(y, x))
    print("Test passed!"if torch.allclose(y, x) else "Test failed!")