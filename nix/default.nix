{
  lib,
  pkgs,
  python3Packages,
}: let
  callPackage = lib.callPackageWith (pkgs // packages // python3Packages);
  packages = {
    dinov2 = callPackage ./dinov2.nix {};
  };
in
  packages
