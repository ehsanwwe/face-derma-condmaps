# models/

Put the BiSeNet checkpoint here (not tracked by git, it is ~51 MB):

    models/79999_iter.pth

It comes from https://github.com/zllrunning/face-parsing.PyTorch (CelebAMask-HQ weights).
`docker compose` mounts this folder read-only into the container at `/app/models`.
