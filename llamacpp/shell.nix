# NixOS runtime for the llama.cpp deployment (llamacpp/run.sh).
#
# The Python sidecars install manylinux wheels (torch, paddle, opencv) into a uv venv.
# Those wheels expect libstdc++, zlib and libGL in the usual FHS places, which NixOS
# does not have, so this shell puts them on LD_LIBRARY_PATH. The NVIDIA driver's
# libcuda comes from /run/opengl-driver. llama-server itself is your own build.
#
#   nix-shell llamacpp/shell.nix --run 'llamacpp/run.sh'
{ pkgs ? import <nixpkgs> { } }:

let
  runtimeLibs = with pkgs; [
    stdenv.cc.cc.lib
    zlib
    libGL
    glib
    libsndfile
  ];
in
pkgs.mkShell {
  name = "interfaze-llamacpp";
  packages = with pkgs; [ python312 uv ffmpeg curl ];
  LD_LIBRARY_PATH = "/run/opengl-driver/lib:" + pkgs.lib.makeLibraryPath runtimeLibs;
  # uv's own CPython builds are linked for an FHS loader that NixOS stubs out.
  UV_PYTHON = "${pkgs.python312}/bin/python3.12";
  UV_PYTHON_DOWNLOADS = "never";
}
