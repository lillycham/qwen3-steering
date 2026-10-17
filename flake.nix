{
  description = "Train a few MLP neurons of Qwen3-0.6B into an owl, then move the deltas to the base model";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

  outputs = { self, nixpkgs }:
    let
      systems = [ "aarch64-darwin" "x86_64-darwin" "aarch64-linux" "x86_64-linux" ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
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
        });
    };
}
