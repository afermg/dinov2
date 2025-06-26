{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    nixpkgs_master.url = "github:NixOS/nixpkgs/master";
    systems.url = "github:nix-systems/default";
    flake-utils.url = "github:numtide/flake-utils";
    flake-utils.inputs.systems.follows = "systems";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      systems,
      ...
    }@inputs:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          system = system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };

      in
      with pkgs;
      rec {
        # py311 = (
        #     pkgs.python312
        # pkgs.python312.override {
        # packageOverrides = _: super: {
        #   scikit-learn = super.scikit-learn.overridePythonAttrs (old: rec {
        #     version = "1.2.2";
        #     # skip checks, as a few fail but they are irrelevant
        #     doCheck = false;
        #     src = super.fetchPypi {
        #       pname = "scikit-learn";
        #       inherit version;
        #       hash = "sha256-hCmuow7CTnqMftij+mITrfOBSm776gnhbgoMceGho9c=";
        #     };
        #   });
        #   };
        # }
        packages = {
          dinov2 = pkgs.python312.pkgs.callPackage ./nix/dinov2.nix { };
        };
        devShells = {
          default =
            let
              python_with_pkgs = (
                python312.withPackages (pp: [
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
                runHook venvShellHook
                export PYTHONPATH=${python_with_pkgs}/${python_with_pkgs.sitePackages}:$PYTHONPATH
              '';
            };
        };
      }
    );
}
