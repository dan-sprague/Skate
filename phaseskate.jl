#!/usr/bin/env -S julia -t5
# Activate the project from the script's own directory
using Pkg
Pkg.activate(@__DIR__)

using PhaseSkate, Tachikoma
PhaseSkate.app()
