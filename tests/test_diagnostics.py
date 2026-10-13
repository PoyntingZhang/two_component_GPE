import csv
import numpy as np
import pytest
import matplotlib

matplotlib.use("Agg")
cp=pytest.importorskip("cupy")
try:
    if cp.cuda.runtime.getDeviceCount()==0:
        pytest.skip("CUDA GPU required",allow_module_level=True)
except cp.cuda.runtime.CUDARuntimeError:
    pytest.skip("CUDA GPU required",allow_module_level=True)

from gpe.core.grids import get_grids
from gpe.core.helper import get_N
from gpe.io.diagnostics import get_E, get_L, is_due
from gpe.io.nk_extraction import nk_extraction
from gpe.io.export import export
from gpe.io.monitor import monitor
import importlib

monitor_module=importlib.import_module("gpe.io.monitor")


def make_state(n=8,length=2*np.pi):
    x,k=get_grids(n,length)
    rng=np.random.default_rng(42)
    psi=cp.asarray(rng.normal(size=(n,n,n))+1j*rng.normal(size=(n,n,n)))
    return psi,x,k


def test_scalar_plane_wave():
    n,length=16,2*np.pi
    x,k=get_grids(n,length)
    phase=2*x[:,None,None]-3*x[None,:,None]+x[None,None,:]
    psi=cp.exp(1j*phase)/length**1.5
    N=float(get_N(psi,x))
    E=float(get_E(cp.fft.fftn(psi),x,k))
    assert N==pytest.approx(1,abs=1e-12)
    assert E==pytest.approx(14,abs=1e-11)


@pytest.mark.parametrize("charge",[-1,1])
def test_vortex_angular_momentum(charge):
    x,k=get_grids(48,16)
    X,Y,Z=x[:,None,None],x[None,:,None],x[None,None,:]
    psi=(X+charge*1j*Y)*cp.exp(-(X*X+Y*Y+Z*Z)/2)
    N=float(get_N(psi,x))
    L=float(get_L(psi,x,k,slab_size=7))
    assert L==pytest.approx(charge*N,rel=1e-9,abs=1e-10)


@pytest.mark.parametrize("n",[2,7,8,18])
def test_shells_and_parseval(n):
    psi,x,k=make_state(n)
    F=cp.fft.fftn(psi)
    spectrum={key:cp.asnumpy(value) for key,value in nk_extraction(F,x,k,slab_size=3).items()}
    dx=float(x[1]-x[0])
    dk=abs(float(k[1]-k[0]))
    radius=np.sqrt(sum(axis**2 for axis in np.meshgrid(cp.asnumpy(k),cp.asnumpy(k),
                                                     cp.asnumpy(k),indexing="ij")))
    bins=np.floor(radius/dk+0.5).astype(int)
    counts=np.bincount(bins.ravel(),minlength=len(spectrum['k']))
    power=abs(np.fft.fftn(cp.asnumpy(psi)))**2
    population=np.bincount(bins.ravel(),weights=power.ravel(),minlength=len(counts))*dx**3/psi.size
    np.testing.assert_array_equal(spectrum['counts'],counts)
    np.testing.assert_allclose(spectrum['Nk'],population,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(spectrum['nk']*counts*dk**3,population,rtol=1e-12,atol=1e-12)
    assert spectrum['Nk'].sum()==pytest.approx(float(get_N(psi,x)),rel=1e-12)
    assert counts.sum()==n**3
    assert np.isfinite(spectrum['nk']).all()
    assert spectrum['edges'][0]==0
    np.testing.assert_array_equal(spectrum['complete_shells'],spectrum['edges'][1:]<=n*dk/2)


def test_zero_mode_spectrum():
    x,k=get_grids(8,2*np.pi)
    psi=cp.ones((8,8,8),dtype=cp.complex128)/(2*np.pi)**1.5
    spectrum=nk_extraction(cp.fft.fftn(psi),x,k)
    assert float(spectrum['Nk'][0])==pytest.approx(1)
    np.testing.assert_allclose(cp.asnumpy(spectrum['Nk'][1:]),0,atol=1e-25)
    assert int(spectrum['counts'][0])==1


def test_intervals_and_export_roundtrip(tmp_path,monkeypatch):
    psi,x,k=make_state()
    output=tmp_path/"output"
    original_fft=cp.fft.fftn
    calls=[]
    def tracked_fft(*args,**kwargs):
        calls.append(1)
        return original_fft(*args,**kwargs)
    monkeypatch.setattr(cp.fft,"fftn",tracked_fft)
    export(0,psi,k,x,psi_interval=1,nk_interval=1,N_interval=1,output_dir=output)
    export(1,psi,k,x,output_dir=output)
    assert not output.exists()
    for i in range(1,7):
        export(i,psi,k,x,psi_interval=2,nk_interval=3,N_interval=2,output_dir=output)
    assert len(calls)==2
    assert sorted(p.name for p in output.glob("psi_*.npy"))==[
        "psi_00000002.npy","psi_00000004.npy","psi_00000006.npy"]
    np.testing.assert_array_equal(np.load(output/"psi_00000002.npy"),cp.asnumpy(psi))
    with np.load(output/"psi_00000002_grid.npz") as metadata:
        np.testing.assert_array_equal(metadata['x_array'],cp.asnumpy(x))
        np.testing.assert_array_equal(metadata['k_array'],cp.asnumpy(k))
        assert metadata['N']==pytest.approx(float(get_N(psi,x)))
    with np.load(output/"nk_00000003.npz") as spectrum:
        assert spectrum['Nk'].sum()==pytest.approx(float(get_N(psi,x)))
    with (output/"N.csv").open() as file:
        rows=list(csv.DictReader(file))
    assert [int(row['i']) for row in rows]==[2,4,6]
    assert all(float(row['N'])==pytest.approx(float(get_N(psi,x))) for row in rows)


def test_monitor_images_and_fft_sharing(monkeypatch,capsys):
    psi,x,k=make_state()
    original=cp.asnumpy(psi).copy()
    F=cp.fft.fftn(psi)
    images=[]
    monkeypatch.setattr(monitor_module,"_show_image",
                        lambda data,*args,**kwargs: images.append(cp.asnumpy(data)))
    monkeypatch.setattr(monitor_module.plt,"show",lambda: None)
    def no_fft(*args,**kwargs):
        raise AssertionError("The supplied FFT should be reused")
    monkeypatch.setattr(cp.fft,"fftn",no_fft)
    result=monitor(6,psi,k,x,show_nk_interval=3,show_TOF_image_interval=2,
                   show_verticle_TOF_interval=2,show_TOF_slice_interval=2,
                   show_verticle_TOF_slice_interval=2,show_magnifier_interval=2,
                   show_magnifier_slice_interval=2,show_phase_interval=2,psi_k=F)
    assert len(images)==7
    dx=float(x[1]-x[0])
    dk=float(k[1]-k[0])
    density_k=abs(np.fft.fftn(original))**2*dx**3/(psi.size*dk**3)
    np.testing.assert_allclose(images[0],np.fft.fftshift(density_k.sum(axis=2))*dk)
    np.testing.assert_allclose(images[1],np.fft.fftshift(density_k.sum(axis=0))*dk)
    np.testing.assert_allclose(images[2],np.fft.fftshift(density_k[:,:,0]))
    np.testing.assert_allclose(images[3],np.fft.fftshift(density_k[0,:,:]))
    np.testing.assert_allclose(images[4],(abs(original)**2).sum(axis=2)*dx)
    np.testing.assert_allclose(images[5],abs(original[:,:,4])**2)
    np.testing.assert_allclose(images[6],np.angle(original[:,:,4]))
    np.testing.assert_array_equal(cp.asnumpy(psi),original)
    assert result['spectrum']['Nk'].sum()==pytest.approx(float(get_N(psi,x)))
    assert capsys.readouterr().out==""
    assert monitor_module.plt.get_fignums()==[]


def test_monitor_disabled_and_scalar_output(monkeypatch,capsys):
    psi,x,k=make_state()
    monkeypatch.setattr(monitor_module.plt,"show",lambda: None)
    assert monitor(1,psi,k,x)=={}
    assert monitor(0,psi,k,x,print_scalar_interval=1)=={}
    result=monitor(2,psi,k,x,print_scalar_interval=2)
    assert set(result)=={"N","E","L"}
    assert all(np.isfinite(value) for value in result.values())
    assert "E_kin=" in capsys.readouterr().out


def test_norm_only_export_and_supplied_fft(tmp_path,monkeypatch):
    psi,x,k=make_state()
    F=cp.fft.fftn(psi)
    def no_fft(*args,**kwargs):
        raise AssertionError("No FFT is needed here")
    monkeypatch.setattr(cp.fft,"fftn",no_fft)
    export(1,psi,k,x,N_interval=1,output_dir=tmp_path)
    export(2,psi,k,x,nk_interval=2,output_dir=tmp_path,psi_k=F)
    assert (tmp_path/"N.csv").exists()
    assert (tmp_path/"nk_00000002.npz").exists()


def test_image_orientation_and_phase_mask(monkeypatch):
    psi,x,k=make_state()
    psi[:,:,4]=0
    shown=[]
    def capture():
        fig=monitor_module.plt.gcf()
        shown.append(np.asarray(fig.axes[0].images[0].get_array()).copy())
    monkeypatch.setattr(monitor_module.plt,"show",capture)
    monitor(1,psi,k,x,show_magnifier_interval=1,show_phase_interval=1)
    expected=cp.asnumpy((abs(psi)**2).sum(axis=2))*float(x[1]-x[0])
    np.testing.assert_allclose(shown[0],expected.T)
    assert np.isnan(shown[1]).all()
    assert monitor_module.plt.get_fignums()==[]


@pytest.mark.parametrize("i,interval,expected",[(0,1,False),(1,0,False),(2,None,False),
                                               (1,2,False),(2,2,True),(6,3,True)])
def test_schedule(i,interval,expected):
    assert is_due(i,interval)==expected
