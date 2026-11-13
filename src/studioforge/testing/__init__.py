"""Test and staging facilities shipped with StudioForge (D85).

Nothing in here is used by a normal install. :mod:`studioforge.testing.stub_llama_server`
is a stand-in for ``llama-server`` that the unit suite and a no-GPU staging
instance share; it is reachable only through ``engine.stub`` (refused unless
the GPU probe is the fake one) or by running the module by hand.
"""
