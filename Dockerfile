# CPU image for nllb-server. Build from the repo root:
#   docker build -t nllb-server .
#   docker run -p 8080:8080 -v $PWD/models:/models nllb-server
# The model is mounted, not baked in: 1.2 GB layers make every push slow.

FROM ubuntu:24.04 AS build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake git ca-certificates \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
# GGML_NATIVE=OFF: the image must run on whatever CPU the cloud gives you, not the build machine's.
RUN cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DBUILD_SHARED_LIBS=OFF -DNLLB_GGML_PATCHES=OFF \
 && cmake --build build -j --target nllb-server

FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/build/bin/nllb-server /usr/local/bin/
EXPOSE 8080
HEALTHCHECK CMD curl -fs http://localhost:8080/health || exit 1
ENTRYPOINT ["nllb-server", "--host", "0.0.0.0", "--port", "8080", "-m", "/models/nllb-600M-f16.gguf"]
