import numpy as np
import torch


def code2img(code):
    # type: (torch.Tensor) -> np.ndarray

    if len(code.shape) == 4 and code.shape[0] == 1:
        code = code[0]

    # shape=(H, W, C) and values in [-1, 1]
    code = code.cpu().numpy().transpose((1, 2, 0))

    # values in [0, 255]
    code = (0.5 * (code + 1)) * 255
    code = code.astype(np.uint8)

    h, w, nc = code.shape
    codes = []
    out = None
    for a in range(0, nc, 3):
        code_chunk = code[:, :, a:a + 3]
        cc = code_chunk.shape[-1]
        if cc == 1:
            code_chunk = np.concatenate([
                code_chunk, code_chunk, code_chunk
            ], -1)
        elif cc == 2:
            z = np.zeros((code_chunk.shape[0], code_chunk.shape[1], 1))
            code_chunk = np.concatenate([code_chunk, z], -1)

        out = code_chunk if out is None else np.hstack((out, code_chunk))

        codes.append(code_chunk)

    return out


def main():
    code = torch.zeros((1, 6, 4, 4))
    out = code2img(code)
    print(out.shape)


if __name__ == '__main__':
    main()
