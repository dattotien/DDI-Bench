# Data layout

Mỗi dataset có thư mục riêng — không dataset nào ghi đè dataset nào nữa.
Mọi thông số (`num_ent`, `num_rel`, `task`, tên file) khai báo tại
[`../dataset_registry.py`](../dataset_registry.py).

```
data/
  <dataset>_cluster/      train.txt, valid_S{0,1,2}.txt, test_S{0,1,2}.txt
  <dataset>_random/       (idem)
  initial/<dataset>/      DB_molecular_feats.pkl, id2smiles.json, relations_2hop.txt
```

Chọn dataset khi chạy: `python main.py --model MLP --dataset mecddi --dataset_type cluster`

Kiểm tra data trước khi train: `python check_data.py` (hoặc `python check_data.py mecddi cluster`).

## Tình trạng hiện tại

| dataset    | num_ent | num_rel | task       | cluster | random | initial/                                       |
|------------|---------|---------|------------|---------|--------|------------------------------------------------|
| `drugbank` | 1710    | 86      | multiclass | ✅      | ✅     | ❌ — đã xoá, chỉ chạy được MSTE (`--use_feat 0`) |
| `mecddi`   | 1567    | 103     | multiclass | ✅      | ❌     | feats + id2smiles đủ 1567 drug, **không có** relations_2hop |
| `mudi`     | 1295    | 4       | multiclass | ❌      | ❌     | ❌ (copy vào trước khi chạy)                   |
| `twosides` | 645     | 209     | multilabel | ✅      | ✅     | feats + cid2id + cid2smiles + relations_2hop   |

`data/initial/drugbank/` đã được xoá khỏi repo (feature pickle 17 MB +
`relations_2hop.txt` 25 MB). Split của DrugBank vẫn còn, nên muốn chạy lại
DrugBank thì chỉ cần thả 3 file đó vào `data/initial/drugbank/` — registry và
`check_data.py` sẽ báo chính xác file nào còn thiếu.

## MecDDI

`data/initial/mecddi/` phủ đủ **1567/1567** drug.

- `DB_molecular_feats.pkl` đánh index theo cột **`Node_ID`**, *không* theo thứ tự
  dòng (khác DrugBank). Registry khai báo `feat_id_key='Node_ID'` nên code tự sắp
  lại theo drug id. **Code cũ lấy theo thứ tự dòng → mọi kết quả MecDDI chạy
  trước đây đều dùng fingerprint của thuốc khác; cần chạy lại.**
- `node2drugbank.json` giữ ánh xạ `node id -> DrugBank ID` cho cả 1567 id.
- Không có `relations_2hop.txt` → Decagon / TIGER không chạy được.
- Trong 1472 drug gốc có 238 dòng mà `Morgan_Features` không khớp với `SMILES`
  của chính nó — hai cột đến từ hai bản DrugBank khác nhau (không phải lệch
  dòng). Muốn đồng nhất một nguồn thì chạy `build_features.py --mode rebuild`,
  nhưng nó đổi feature của ~21% drug nên sẽ đổi luôn kết quả benchmark.

### Sinh lại feature từ CSV `Drugbank_ID,SMILES`

```
python build_features.py --dataset mecddi --csv ../../drugbank_smiles_map.csv          # dry run
python build_features.py --dataset mecddi --csv ../../drugbank_smiles_map.csv --write  # ghi thật (+ .bak)
```

Fingerprint dùng `GetHashedMorganFingerprint(mol, radius=2, nBits=1024)` (count
vector) — công thức này tái tạo đúng từng bit feature có sẵn, nên feature mới
cùng hệ với feature cũ.

CSV không có node id. Script lấy id từ `node_map_file` khai báo trong registry
(`node2drugbank.json`); nếu dataset chưa có node map thì nó dò theo thứ tự
DrugBank ID tăng dần và bỏ qua các chỗ mơ hồ. Đừng dùng `--ambiguous guess`:
với MecDDI cách đoán đó gán sai 4/7 drug so với node map thật.

## Định dạng file split

- **multiclass** (`drugbank`, `mecddi`, `mudi`): mỗi dòng `head tail rel`
- **multilabel** (`twosides`): mỗi dòng `head tail r0,r1,...,rN positive_flag`

## `initial/<dataset>/` cần gì

| file                     | model dùng tới                     | bắt buộc?                    |
|--------------------------|------------------------------------|------------------------------|
| `DB_molecular_feats.pkl` | mọi model có `--use_feat 1`        | có (trừ MSTE)                |
| `id2smiles.json`         | SSI-DDI, SA-DDI, SAGAN, TIGER      | chỉ khi chạy các model đó    |
| `relations_2hop.txt`     | Decagon, TIGER                     | chỉ khi chạy Decagon / TIGER |
| `node2drugbank.json`     | chỉ `build_features.py`            | không                        |

Dataset nào thiếu file sẽ báo lỗi rõ ràng ngay lúc load (`FileNotFoundError`
kèm đường dẫn mong đợi), thay vì chạy nhầm sang data của dataset khác. Riêng
drug thiếu SMILES thì SSI-DDI / SA-DDI thay bằng phân tử rỗng và in cảnh báo,
còn drug thiếu feature nhận vector 0 kèm cảnh báo.

## Thêm dataset mới

1. Bỏ data vào `data/<tên>_cluster/` (+ `_random/`) và `data/initial/<tên>/`.
2. Thêm một entry vào `DATASETS` trong `../dataset_registry.py`.

Không cần sửa gì trong `trainer.py`, `data_process.py`, `utils.py` hay `models/`
— code branch theo `args.task` (`multiclass` / `multilabel`), không theo tên dataset.

## Cache

TIGER ghi cache subgraph/molecule vào chính thư mục split
(`data/<dataset>_<type>/tiger_mol_sp.json`, `data/<dataset>_<type>/rw/`),
nên cache của các dataset cũng tách bạch. Xoá thư mục cache nếu bạn thay data.
