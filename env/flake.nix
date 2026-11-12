{
  description = "Python environment for the binary-docking-gates measurements on SmolLM2";

  # Kept in its own directory and used as `path:./env`, so nix copies only this
  # folder (not the multi-GB data next to it) and does not resolve the stray ~/.git.
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

  outputs = { self, nixpkgs }:
    let
      system = "aarch64-darwin";
      pkgs = nixpkgs.legacyPackages.${system};
      python = pkgs.python3.withPackages (p: [ p.numpy p.pyarrow p.tokenizers p.torch ]);
    in {
      packages.${system}.default = python;
      devShells.${system}.default = pkgs.mkShell { packages = [ python ]; };
    };
}
