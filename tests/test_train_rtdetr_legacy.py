"""Carga de checkpoints de RT-DETR guardados con una versión vieja de `transformers`.

El checkpoint legacy usa `encoder.encoder.*`, `fc1`/`fc2` sueltos y `self_attn.out_proj`; la versión
instalada espera `encoder.aifi.*`, `mlp.fc1`/`mlp.fc2` y `self_attn.o_proj`. Si una versión nueva de
transformers vuelve a renombrar, estas pruebas fallan en vez de dejar capas reinicializadas al azar."""
import re

import pytest

pytest.importorskip("torch")
pytest.importorskip("safetensors")
transformers = pytest.importorskip("transformers")

import torch  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

from alpr.train.train_rtdetr import _load_base_model, _remap_legacy_rtdetr_state_dict  # noqa: E402


def _to_legacy_name(key):
    key = key.replace("model.encoder.aifi.", "model.encoder.encoder.")
    key = key.replace(".self_attn.o_proj.", ".self_attn.out_proj.")
    return re.sub(r"(\.layers\.\d+)\.mlp\.(fc[12])\.", r"\1.\2.", key)


def _to_current_name(key):
    return _remap_legacy_rtdetr_state_dict({key: None}).popitem()[0]


def _tiny_model():
    from transformers import RTDetrConfig, RTDetrForObjectDetection, RTDetrResNetConfig
    backbone = RTDetrResNetConfig(embedding_size=16, hidden_sizes=[16, 32, 64, 128], depths=[1, 1, 1, 1],
                                  out_features=["stage2", "stage3", "stage4"])
    cfg = RTDetrConfig(
        backbone_config=backbone, encoder_hidden_dim=32, encoder_in_channels=[32, 64, 128], encoder_layers=1,
        encoder_ffn_dim=64, encoder_attention_heads=2, decoder_layers=1, decoder_ffn_dim=64,
        decoder_attention_heads=2, decoder_in_channels=[32, 32, 32], d_model=32, num_queries=10,
        num_labels=1, id2label={0: "plate"}, label2id={"plate": 0}, num_denoising=0, decoder_n_points=2,
        hidden_expansion=0.5, anchor_image_size=None)
    return RTDetrForObjectDetection(cfg)


def test_remap_traduce_los_nombres_viejos_y_deja_los_demas():
    old = {
        "model.encoder.encoder.0.layers.0.self_attn.out_proj.weight": 1,
        "model.encoder.encoder.0.layers.0.fc1.weight": 2,
        "model.encoder.encoder.0.layers.0.fc2.bias": 3,
        "model.decoder.layers.0.fc1.weight": 4,
        "model.decoder.layers.0.self_attn.out_proj.bias": 5,
        "model.backbone.model.embedder.weight": 6,
        "class_embed.0.weight": 7,
    }
    new = _remap_legacy_rtdetr_state_dict(old)
    assert new == {
        "model.encoder.aifi.0.layers.0.self_attn.o_proj.weight": 1,
        "model.encoder.aifi.0.layers.0.mlp.fc1.weight": 2,
        "model.encoder.aifi.0.layers.0.mlp.fc2.bias": 3,
        "model.decoder.layers.0.mlp.fc1.weight": 4,
        "model.decoder.layers.0.self_attn.o_proj.bias": 5,
        "model.backbone.model.embedder.weight": 6,
        "class_embed.0.weight": 7,
    }


def test_remap_es_idempotente_con_nombres_actuales():
    current = {"model.encoder.aifi.0.layers.0.mlp.fc1.weight": 1,
               "model.encoder.aifi.0.layers.0.self_attn.o_proj.weight": 2}
    assert _remap_legacy_rtdetr_state_dict(current) == current


@pytest.mark.parametrize("rename", [_to_legacy_name, _to_current_name], ids=["nombres_viejos", "nombres_nuevos"])
def test_checkpoint_se_carga_completo_y_con_los_pesos_originales(tmp_path, capsys, rename):
    torch.manual_seed(0)
    original = _tiny_model()
    original.save_pretrained(tmp_path)
    expected = {k: v.clone() for k, v in original.state_dict().items()}

    # Reescribe el checkpoint con los nombres de cada formato (viejo o nuevo).
    st_path = tmp_path / "model.safetensors"
    renamed = {rename(k): v.contiguous() for k, v in load_file(str(st_path)).items()}
    save_file(renamed, str(st_path), metadata={"format": "pt"})

    torch.manual_seed(1)  # otra semilla: lo que no se cargue quedaría distinto del original
    loaded = _load_base_model(str(tmp_path), type(original))

    assert "sin faltantes" in capsys.readouterr().out
    for name, tensor in loaded.state_dict().items():
        assert torch.equal(tensor, expected[name]), f"peso distinto tras cargar el checkpoint: {name}"


def test_los_dos_formatos_de_nombres_son_distintos():
    """Guarda contra un test vacío: si los dos renombrados dieran lo mismo, el parametrizado no probaría nada."""
    key = "model.encoder.aifi.0.layers.0.mlp.fc1.weight"
    assert _to_legacy_name(key) != _to_current_name(key)
