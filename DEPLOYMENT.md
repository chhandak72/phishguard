# Deployment guide — simple, free options

This file explains several easy, low-cost (free-tier) ways to deploy PhishGuard so you can run it for demos or light usage. Pick the option that sounds easiest for you. All options assume your project is tracked in a Git repository (GitHub is recommended).

Important notes

- If you include the `data/` CSV files in the repository, make sure no single file is >100 MB (GitHub hard limit). For large files, use Git LFS or host the datasets externally (Dropbox/S3/Gist).
- The backend trains a model if `backend/models/stacking_pipeline.joblib` is missing. The included `backend/render_startup.sh` trains on deploy if the model is not present.

## Recommended (easiest): Frontend on Vercel, Backend on Render (both have free tiers)

Why: Vercel will host the React app with zero config. Render can run the Python FastAPI backend and will execute `render_startup.sh` to train the model the first time (or reuse a persisted model volume).

Steps (one-time)

1. Push your repo to GitHub (see "Including datasets" below).
2. Backend (Render)
   - Sign up at https://render.com (free tier) and connect your GitHub repository.
   - Create a new Web Service from your repo. Set the root to `backend` (or use the provided `render.yaml` blueprint).
   - Build command: `pip install -r requirements.txt && chmod +x render_startup.sh`
   - Start command: `bash render_startup.sh`
   - Set environment variables in Render (optional):
     - `CORS_ORIGINS` → e.g. `https://<your-vercel-app>.vercel.app`
     - `PYTHON_VERSION` → `3.11.0` (recommended)
   - Set a persistent disk (Render free tier supports small disk volumes) and mount it at `/opt/render/project/src/models` so the trained model file persists between deploys.
   - Deploy. The first deploy will train the model (can take a few minutes). The `render_startup.sh` script will save `backend/models/stacking_pipeline.joblib` into the mounted models directory.

3. Frontend (Vercel)
   - Sign up at https://vercel.com and connect your GitHub repo.
   - Add a new Project → import your repository.
   - Configure the build settings:
     - Root (if asked): leave empty (we will set the build to run the frontend folder)
     - Build command: `npm --prefix frontend run build`
     - Output Directory: `frontend/dist`
   - Set an environment variable (on Vercel):
     - `VITE_API_URL` → `https://<your-render-backend-url>` (the public URL Render gives your backend)
   - Deploy. Vercel will build and host your frontend at `https://<your-vercel-app>.vercel.app`.

How the pieces connect

- Frontend sends requests to the path `/predict` via the Vite proxy in development. In production we set `VITE_API_URL` at build time so the client calls your Render backend directly.

## Alternative single-host option (Render only)

If you prefer to keep everything on Render, you can create two services in the same Render account:

- Backend service: same as above (root `backend`).
- Frontend static site (Static Site service): set the root to `frontend`, build command `npm --prefix frontend run build`, publish directory `frontend/dist`. Set `VITE_API_URL` to your backend URL under the static site environment variables before building.

## Alternative (one container): Docker Compose / Docker Hub

The repo already contains a `docker-compose.yml`. You can build locally and run both services with Docker Compose. This is great for local demos but requires a VPS to run in production (e.g., a free or cheap VM provider). Not covered in detail here.

## Including datasets in GitHub

If you want your hosted backend to train on the same CSV datasets, add the `data/` folder to the repo and push to GitHub. Steps:

1. Check file sizes:

```bash
du -sh data/*
```

2. If any file is >50 MB, use Git LFS (recommended for large CSVs):

```bash
git lfs install
git lfs track "data/*.csv"
git add .gitattributes
git add data/*.csv
git commit -m "Add datasets (via LFS)"
git push origin main
```

3. If files are small (<100MB each), you can simply add and push:

```bash
git add data/
git commit -m "Add training datasets"
git push origin main
```

Notes

- Training on deploy will incur CPU time on Render; it may take several minutes.
- If you want instant deployments without training latency, train locally and commit `backend/models/stacking_pipeline.joblib` (or upload the artifact to an object store and have `render_startup.sh` download it during boot).

Troubleshooting

- If the backend returns `503 Model not loaded.`, check Render logs — training may still be in progress or failed due to missing dependencies.
- If a CSV file is missing in the deployed environment, confirm it exists in the repo path `backend/data/` (render working dir may differ; the `train.py` resolves data paths relative to the backend root).

## Quick checklist to get running now

1. Create a GitHub repo and push your project.
2. On Render: create backend service (root `backend`) — let it train and start.
3. On Vercel: import repo, set build command and output directory as above, set `VITE_API_URL` to your Render URL, deploy.

If you'd like, I can:

- Create a ready-to-push GitHub workflow (GitHub Actions) to build frontend and upload `frontend/dist` to the `gh-pages` branch (publish frontend to GitHub Pages) instead of Vercel.
- Create a tiny script to upload the trained model to an S3-like object store and have the backend download it at startup (faster deploys).

I added automation to this repo to make deployment smoother:

- `.gitattributes` — configures Git LFS for `data/*.csv` so you can safely commit datasets.
- `scripts/train_and_commit.sh` — convenience script that trains the model locally (sample mode for speed) and commits the produced `backend/models/stacking_pipeline.joblib` into git so the backend will not need to train on deploy.
- `.github/workflows/deploy-frontend-gh-pages.yml` — GitHub Actions workflow which will build the frontend and publish `frontend/dist` to the `gh-pages` branch whenever you push to `main`. This gives you a free frontend hosting option (GitHub Pages) that avoids needing a separate provider for the static site.

Suggested immediate steps to complete Render-only deployment (minimal waiting):

1. Train locally and commit model (optional but recommended to avoid training time on Render):

```bash
# from repo root
chmod +x scripts/train_and_commit.sh
./scripts/train_and_commit.sh
git push origin main
```

2. Push repo to GitHub (if not already pushed). The GH Actions workflow will publish the frontend to `gh-pages` automatically.

3. On Render (only backend required):

- Create a new Web Service using this repo, root `backend`.
- Build command: `pip install -r requirements.txt && chmod +x render_startup.sh`
- Start command: `bash render_startup.sh`
- Attach a persistent disk and mount to `/opt/render/project/src/models` (so the model file persists across redeploys).

4. If you prefer Render static site instead of GitHub Pages, use the `render.yaml` blueprint included which defines both services (backend + frontend). After creating services Render will build and publish both.

---

If you want me to add a GitHub Actions workflow for automatic frontend publishing or to prepare a push that adds `data/` (with LFS config) to the repo, tell me which and I'll create the files and run the necessary changes.
