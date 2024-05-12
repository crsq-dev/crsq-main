from crsq.blocks.antisymmetrization import AntisymmetrizationSpec
import crsq.blocks.energy_initialization as ene_ini
import crsq.blocks.wave_function as wf
import crsq.heap as heap

def test_energy_state_bits_preparation_block():
    """ EnergyStateBitsPreparationBlock """
    outer = heap.Frame()
    energy_config_weights = [1.0]
    t = ene_ini.EnergyConfigBitsPreparationBlock(energy_config_weights)
    outer.invoke(t.bind(p=None))

def make_elec_orbitals(dim, nbits, num_conf, ne, use_spins=False):
    """ make orbitals for electrons """
    configs = []
    N=2**nbits
    for c in range(num_conf):
        electrons = []
        for i in range(ne):
            dims = []
            for d in range(dim):
                amps = []
                for k in range(N):
                    amps.append(k/N)
                dims.append(amps)
            if use_spins:
                spins = [1, 0]
                dims.append(spins)
            electrons.append(dims)
        configs.append(electrons)
    return configs

def make_nucl_orbitals(dim, nbits, num_conf, ne):
    """ make orbitals for nuclei """
    configs = []
    N=2**nbits
    for c in range(num_conf):
        nuclei = []
        for i in range(ne):
            dims = []
            for d in range(dim):
                amps = []
                for k in range(N):
                    amps.append(k/N)
                dims.append(amps)
            nuclei.append(dims)
        configs.append(nuclei)
    return configs

def test_single_energy_state_initialization_block():
    """ test single energy state initialization """
    outer = heap.Frame()
    num_conf = 1
    dim = 1
    ne = 1
    nn = 1
    nbits = 3
    L=2.0
    wfspec = wf.WaveFunctionRegisterSpec(dim, nbits, L, ne, nn, 0)
    elec_orbitals = make_elec_orbitals(dim, nbits, num_conf, ne)
    nucl_orbitals = make_nucl_orbitals(dim, nbits, num_conf, nn)
    ene_spec = ene_ini.EnergyConfigurationSpec([1.0], elec_orbitals, nucl_orbitals)
    asy_spec = AntisymmetrizationSpec(wfspec, 2)
    state_idx = 0
    t = ene_ini.SlaterDeterminantPreparationBlock(
        ene_spec, asy_spec, state_idx)

    eregs = wfspec.allocate_elec_registers()
    nregs = wfspec.allocate_nucl_registers()
    bregs = asy_spec.allocate_sigma_regs()
    ancilla = asy_spec.allocate_ancilla_register()
    flat_eregs = heap.Binding.flatten(eregs)
    flat_nregs = heap.Binding.flatten(nregs)
    outer.circuit.add_register(*flat_eregs, *flat_nregs)
    outer.invoke(t.bind(eregs=eregs, nregs=nregs, bregs=bregs, shuffle=ancilla))

if __name__ == '__main__':
    test_energy_state_bits_preparation_block()
    test_single_energy_state_initialization_block()
