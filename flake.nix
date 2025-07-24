{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
    nixpkgs_master.url = "github:NixOS/nixpkgs/master";
    systems.url = "github:nix-systems/default";
    flake-utils.url = "github:numtide/flake-utils";
    flake-utils.inputs.systems.follows = "systems";
    nahual-flake.url = "github:afermg/nahual";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
    systems,
    ...
  } @ inputs:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {
          system = system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };
      in
        with pkgs; rec {
          formatter = pkgs.alejandra;
          apps.default = {
            type = "app";
            program =
              writeShellApplication {
                name = "app";
                text = ''
                  python server.py "$1"
                '';
              }
              + "/bin/app";
          };
          packages = pkgs.callPackage ./nix {};
          devShells = {
            default = let
              python_with_pkgs = (
                python312.withPackages (pp: [
                  (inputs.nahual-flake.packages.${system}.nahual)
                  packages.dinov2
                ])
              );
            in
              mkShell {
                packages = [
                  python_with_pkgs
                ];
                currentSystem = system;
                venvDir = "./.venv";
                postVenvCreation = ''
                  unset SOURCE_DATE_EPOCH
                '';
                postShellHook = ''
                  unset SOURCE_DATE_EPOCH
                '';
                shellHook = ''
                  export CUDA_PATH=${pkgs.cudaPackages.cudatoolkit}
                  export LD_LIBRARY_PATH=${pkgs.cudaPackages.cudatoolkit}/lib:${pkgs.cudaPackages.cudnn}/lib:$LD_LIBRARY_PATH
                  export NVCC_APPEND_FLAGS="-Xcompiler -fno-PIC"
                  export TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6"
                  export CUDA_NVCC_FLAGS="-O2 -Xcompiler -fno-PIC"
                  runHook venvShellHook
                  # export PYTHONPATH=${python_with_pkgs}/${python_with_pkgs.sitePackages}:$PYTHONPATH
                  # Set PYTHONPATH to only include the Nix packages, excluding current directory
                  export PYTHONPATH=${python_with_pkgs}/${python_with_pkgs.sitePackages}
                  # Ensure current directory is not in Python path
                  export PYTHONDONTWRITEBYTECODE=1
                '';
              };
          };
        }
    );
}
