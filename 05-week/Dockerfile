FROM agrigorev/zoomcamp-model:2025

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /code

# Install compiler gcc-c++ python3-devel (Fix for python version >= 3.14)
RUN apt-get update && apt-get install -y gcc g++ 

ENV PATH="/code/.venv/bin:$PATH"

COPY "pyproject.toml" "uv.lock" ".python-version" ./
RUN uv sync --locked

COPY "predict_docker.py" ./

EXPOSE 9696

ENTRYPOINT ["uvicorn", "predict_docker:app", "--host", "0.0.0.0", "--port", "9696"]