{
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";
  outputs = { nixpkgs, ... }:
    let
      pkgs = nixpkgs.legacyPackages.x86_64-linux;

      # nixpkgs' intel-npu-driver ships only the Level Zero driver; it dlopens
      # libnpu_driver_compiler.so, which is a prebuilt blob from Intel's release.
      # Version must match the system NPU driver (hardware.cpu.intel.npu).
      npu-driver-compiler = pkgs.stdenv.mkDerivation rec {
        pname = "intel-npu-driver-compiler";
        version = "1.28.0";
        src = pkgs.fetchurl {
          url = "https://github.com/intel/linux-npu-driver/releases/download/v${version}/linux-npu-driver-v${version}.20251218-20347000698-ubuntu2404.tar.gz";
          hash = "sha256-CcyiJ9fxh5wKN4XWOSP2xjj+ogaTPWMfvnJiN8CNA8I=";
        };
        sourceRoot = ".";
        nativeBuildInputs = with pkgs; [ dpkg autoPatchelfHook ];
        buildInputs = with pkgs; [ tbb zlib zstd stdenv.cc.cc.lib ];
        unpackPhase = ''
          tar xzf $src --wildcards './intel-driver-compiler-npu_*.deb'
          dpkg-deb -x intel-driver-compiler-npu_*.deb .
        '';
        installPhase = ''
          install -Dm755 usr/lib/x86_64-linux-gnu/libnpu_driver_compiler.so -t $out/lib
        '';
      };
      pythonEnv = pkgs.python3.withPackages (ps: with ps; [
        ps.openvino
        ps.onnx
      ]);
    in {
      devShells.x86_64-linux.default = pkgs.mkShell {
        packages = with pkgs; [
          cargo
          rustc
          clippy
          rustfmt
          rust-analyzer
          pythonEnv
        ];
        nativeBuildInputs = with pkgs; [ pkg-config rustPlatform.bindgenHook ];
        buildInputs = with pkgs; [
          pipewire
          openvino
          vulkan-loader
          level-zero
          libjpeg_turbo
        ];
        RUST_SRC_PATH = "${pkgs.rustPlatform.rustLibSrc}";
        OPENVINO_INSTALL_DIR = "${pkgs.openvino}";
        LD_LIBRARY_PATH = "/run/opengl-driver/lib:${pkgs.openvino}/runtime/lib/intel64:${pkgs.level-zero}/lib:${npu-driver-compiler}/lib";
      };
    };
}
