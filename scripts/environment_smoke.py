"""Check the local training environment without training or modifying the dataset."""
from pathlib import Path
import json
import sys

import torch
import torchvision
import yaml
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def main():
    project = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((project / "configs/datasets.yaml").read_text(encoding="utf-8"))
    detector = yaml.safe_load((project / config["detection"]["data_config"]).read_text(encoding="utf-8"))
    class_map = json.loads(Path(config["detection"]["class_map"]).read_text(encoding="utf-8"))
    assert list(detector["names"].values()) == class_map["classes"]
    assert Path(detector["path"]).resolve() == Path(config["detection"]["root"]).resolve()
    assert all((Path(detector["path"]) / detector[s]).is_dir() for s in ("train", "val", "test"))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    x = torch.arange(1024, device="cuda", dtype=torch.float32)
    result = (x * 2).sum().item()
    assert result == 1047552.0
    boxes = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=torch.float32, device="cuda")
    scores = torch.tensor([0.9, 0.8], device="cuda")
    assert torchvision.ops.nms(boxes, scores, 0.5).tolist() == [0]
    print(f"Python: {sys.version.split()[0]}", flush=True)
    print(f"PyTorch: {torch.__version__}, torchvision: {torchvision.__version__}, CUDA runtime: {torch.version.cuda}", flush=True)
    print(f"CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}, tensor sum: {result}, CUDA NMS: OK", flush=True)
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    for head, expected in (("30vn", 30), ("vina21", 21)):
        spec = config["classification"][head]
        mapping = json.loads(Path(spec["class_map"]).read_text(encoding="utf-8"))
        expected_folders = [item["folder"] for item in mapping["classes"]]
        split_counts = {}
        for split in ("train", "val", "test"):
            dataset = datasets.ImageFolder(Path(spec["images"]) / split, transform=transform)
            assert len(dataset.classes) == expected and dataset.classes == expected_folders
            split_counts[split] = len(dataset)
        loader = DataLoader(datasets.ImageFolder(Path(spec["images"]) / "train", transform=transform),
                            batch_size=8, shuffle=False, num_workers=2, pin_memory=True)
        images, labels = next(iter(loader))
        images, labels = images.to("cuda"), labels.to("cuda")
        torch.cuda.synchronize()
        print(f"{head}: splits={split_counts}, classes={expected}, batch={tuple(images.shape)}, labels={labels.tolist()}, device={images.device}", flush=True)
    print("Detection config: 266 classes and all split directories verified", flush=True)


if __name__ == "__main__":
    main()
