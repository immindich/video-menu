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
      # RVM's ONNX export, and the split model the daemon runs: the network
      # without its full-resolution refiner step (see tools/split_refiner.py).
      rvm-onnx = pkgs.fetchurl {
        url = "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp16.onnx";
        hash = "sha256-ag1c5swXcCYTvlSFWYebRSHtQkz+FN3EjRrKpE1hb2Q=";
      };
      rvm-model = pkgs.runCommand "rvm-mobilenetv3-fp16-core" {
        nativeBuildInputs = [ (pkgs.python3.withPackages (ps: [ ps.onnx ])) ];
      } ''
        mkdir -p $out
        python ${./tools/split_refiner.py} ${rvm-onnx} $out/rvm_mobilenetv3_fp16_core.onnx
      '';

      # MediaPipe's selfie segmenter (landscape, Apache-2.0), converted to
      # OpenVINO IR by tools/convert_selfie.py: OpenVINO can't read one of its
      # custom TFLite ops, and the conversion also makes it take the daemon's
      # f16 planes directly.
      selfie-tflite = pkgs.fetchurl {
        name = "selfie_segmenter_landscape.tflite";
        url = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter_landscape/float16/1/selfie_segmenter_landscape.tflite";
        hash = "sha256-SQ6epzQxPg3hD6DNnjxhM+NupNsrekm96e8Bn3J5a44=";
      };
      selfie-model = pkgs.runCommand "selfie-segmenter-landscape-openvino" {
        nativeBuildInputs = [ (pkgs.python3.withPackages (ps: [ ps.openvino ])) ];
      } ''
        mkdir -p $out
        python ${./tools/convert_selfie.py} ${selfie-tflite} $out/selfie_segmenter_landscape.xml
      '';

      pythonEnv = pkgs.python3.withPackages (ps: with ps; [
        ps.openvino
        ps.onnx
      ]);

      video-menu = pkgs.rustPlatform.buildRustPackage
        (finalAttrs: {
          pname = "video-menu";
          version = "0.1.0";

          nativeBuildInputs = with pkgs; [
            pkg-config
            rustPlatform.bindgenHook
          ];

          buildInputs = with pkgs; [
            pipewire
            openvino
            vulkan-loader
            level-zero
            libjpeg_turbo
          ];

          OPENVINO_INSTALL_DIR = "${pkgs.openvino}";

          src = ./.;
          cargoHash = "sha256-Dm6bs1rgSuP1BntWum2eB96IGauIhroeD5PKJB/ND7I=";
          cargoDepsName = finalAttrs.pname;
        });
    in {
      packages.x86_64-linux.rvm-model = rvm-model;
      packages.x86_64-linux.selfie-model = selfie-model;
      packages.x86_64-linux.default = video-menu;
      
      devShells.x86_64-linux.default = pkgs.mkShell {
        packages = with pkgs; [
          cargo
          rustc
          clippy
          rustfmt
          rust-analyzer
          pythonEnv
          vulkan-tools
          vulkan-validation-layers
          shaderc
        ];

        inputsFrom = [ video-menu ];

        RUST_SRC_PATH = "${pkgs.rustPlatform.rustLibSrc}";
        OPENVINO_INSTALL_DIR = "${pkgs.openvino}";
        VIDEO_MENU_MODEL = "${rvm-model}/rvm_mobilenetv3_fp16_core.onnx";
        VIDEO_MENU_SELFIE_MODEL = "${selfie-model}/selfie_segmenter_landscape.xml";
        # The GUI's windowing (winit) loads the Wayland and xkbcommon libraries
        # at runtime, like wgpu does Vulkan.
        LD_LIBRARY_PATH = "/run/opengl-driver/lib:${pkgs.wayland}/lib:${pkgs.libxkbcommon}/lib:${pkgs.vulkan-loader}/lib:${pkgs.openvino}/runtime/lib/intel64:${pkgs.level-zero}/lib:${npu-driver-compiler}/lib";
      };
    };
}
