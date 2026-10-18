{
  description = "Train a few MLP neurons of Qwen3 into a persona, then move the deltas to the base model";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

  outputs = { self, nixpkgs }:
    let
      systems = [ "aarch64-darwin" "x86_64-darwin" "aarch64-linux" "x86_64-linux" ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});

      # MLX from Apple's PyPI wheels (as in tiny-lm). The nixpkgs build has no Metal support, because Apple's Metal
      # shader compiler can't run in the Nix sandbox. mlx-metal holds the precompiled GPU kernels. Apple silicon only.
      mlxFor = pkgs: ps:
        let
          version = "0.32.2";
          wheel = ps: args: ps.buildPythonPackage (args // { inherit version; format = "wheel"; });
          mlx-metal = wheel ps {
            pname = "mlx-metal";
            src = pkgs.fetchurl {
              url = "https://files.pythonhosted.org/packages/79/ec/34f37376e26d537fadffb99af3a760d6545e37f5e1a30a552baadf237fc5/mlx_metal-${version}-py3-none-macosx_15_0_arm64.whl";
              sha256 = "55a369250d220b2cf10213a87a2ac1b1a420608c5b35b1df4e7147ac8e32f121";
            };
          };
        in
        wheel ps {
          pname = "mlx";
          src = pkgs.fetchurl {
            url = "https://files.pythonhosted.org/packages/f8/c8/6928f4b9ca8f190c7c7a19c0a67920aa1742c62c6e66b05f7a1e21da728c/mlx-${version}-cp314-cp314-macosx_15_0_arm64.whl";
            sha256 = "8fc433e35a7058e30f7a225c39fce06ba41394fe8e9fd67383b70f0b06de398c";
          };
          # mlx/core.so looks for libmlx.dylib in its own mlx/lib, as a pip install would have it.
          postInstall = ''
            cp -r ${mlx-metal}/${ps.python.sitePackages}/mlx/lib $out/${ps.python.sitePackages}/mlx/
          '';
          dontCheckRuntimeDeps = true;
          pythonImportsCheck = [ "mlx.core" ];
        };
    in
    {
      devShells = forAllSystems (pkgs:
        let
          pythonEnv = pkgs.python3.withPackages (ps: [
            ps.torch
            ps.transformers
            ps.safetensors
            ps.huggingface-hub
            ps.pytest
          ]);
          appleSilicon = pkgs.stdenv.hostPlatform.isDarwin && pkgs.stdenv.hostPlatform.isAarch64;
          # Chat with the 4-bit models and the persona adapters: mlx-lm from nixpkgs, built on Apple's MLX wheels.
          mlxEnv = pkgs.python3.withPackages (ps:
            let mlx = mlxFor pkgs ps;
            in [ mlx ((ps.mlx-lm.override { inherit mlx; }).overridePythonAttrs { doCheck = false; dontCheckRuntimeDeps = true; }) ]);  # (sentencepiece: unused by Qwen)
        in
        {
          default = pkgs.mkShell {
            packages = [ pythonEnv ];
            # A fixed path to this Python for editors, because the store path changes on every update.
            shellHook = ''
              ln -sfn ${pythonEnv} .python-env
              export PYTORCH_ENABLE_MPS_FALLBACK=1
            '';
          };
        } // pkgs.lib.optionalAttrs appleSilicon {
          mlx = pkgs.mkShell { packages = [ mlxEnv ]; };
        });
    };
}
