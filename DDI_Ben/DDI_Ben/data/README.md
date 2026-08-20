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

| dataset    | num_ent | num_rel | task       | cluster | random | initial/                                            |
|------------|---------|---------|------------|---------|--------|-----------------------------------------------------|
| `drugbank` | 1710    | 86      | multiclass | ✅      | ✅     | feats + id2smiles + relations_2hop                  |
| `mecddi`   | 1567    | 103     | multiclass | ✅      | ❌     | feats + id2smiles cho 1560/1567 drug, **không có** relations_2hop |
| `mudi`     | 1295    | 4       | multiclass | ❌      | ❌     | ❌ (copy vào trước khi chạy)                        |
| `twosides` | 645     | 209     | multilabel | ✅      | ✅     | feats + cid2id + cid2smiles + relations_2hop        |

## Vấn đề đã biết của MecDDI

`data/initial/mecddi/` phủ **1560 / 1567** drug (trước đây 1472, đã bù thêm 88 từ
`drugbank_smiles_map.csv` bằng `build_features.py`).

- `DB_molecular_feats.pkl` đánh index theo cột **`Node_ID`**, *không* theo thứ tự
  dòng (khác DrugBank). Registry khai báo `feat_id_key='Node_ID'` nên code tự sắp
  lại theo drug id. **Code cũ lấy theo thứ tự dòng → mọi kết quả MecDDI chạy
  trước đây đều dùng fingerprint của thuốc khác.**
- Còn thiếu 7 drug: id `5, 6, 7, 8, 9, 28, 1520`. CSV có ứng viên nhưng không suy
  được id duy nhất (xem phần dưới). 7 drug này nhận vector 0 và làm SSI-DDI /
  SAGAN / TIGER dừng với lỗi rõ ràng.
- Không có `relations_2hop.txt` → Decagon / TIGER không chạy được.
- Trong 1472 drug gốc có 238 dòng mà `Morgan_Features` không khớp với `SMILES`
  của chính nó — hai cột đến từ hai bản DrugBank khác nhau. Không phải lỗi lệch
  dòng; nếu muốn đồng nhất một nguồn thì chạy `build_features.py --mode rebuild`
  (đổi feature của ~21% drug cũ, nên sẽ đổi kết quả benchmark).

### Bù drug thiếu từ file CSV `Drugbank_ID,SMILES`

```
python build_features.py --dataset mecddi --csv ../../drugbank_smiles_map.csv          # dry run
python build_features.py --dataset mecddi --csv ../../drugbank_smiles_map.csv --write  # ghi thật (+ .bak)
```

CSV không có node id, nhưng node id được đánh theo đúng thứ tự DrugBank ID tăng
dần nên script khớp danh sách CSV đã sắp xếp với các node id đã biết để suy ra id
còn thiếu. Fingerprint dùng `GetHashedMorganFingerprint(mol, radius=2, nBits=1024)`
— công thức này tái tạo đúng từng bit feature có sẵn, nên feature mới cùng hệ với
feature cũ.

7 drug còn lại nằm trong 3 khoảng mà CSV có nhiều ứng viên hơn số id thiếu:

| node id  | ứng viên trong CSV                                     |
|----------|--------------------------------------------------------|
| 5..9     | DB00016, DB00017, DB00019, DB00024, DB00026, DB00030 (6 chọn 5) |
| 28       | DB00107, DB00109 (2 chọn 1)                            |
| 1520     | DB14158, DB14159 (2 chọn 1)                            |

Muốn xử lý dứt điểm thì cần danh sách drug gốc của MecDDI (hoặc script đã sinh ra
`DB_molecular_feats.pkl` ban đầu). Tạm chấp nhận đoán theo thứ tự thì thêm
`--ambiguous guess`, nhưng đoán sai nghĩa là gán nhầm phân tử cho drug id.

## Định dạng file split

- **multiclass** (`drugbank`, `mecddi`, `mudi`): mỗi dòng `head tail rel`
- **multilabel** (`twosides`): mỗi dòng `head tail r0,r1,...,rN positive_flag`

## `initial/<dataset>/` cần gì

| file                     | model dùng tới                | bắt buộc?                          |
|--------------------------|-------------------------------|------------------------------------|
| `DB_molecular_feats.pkl` | mọi model có `--use_feat 1`   | có (trừ MSTE)                      |
| `id2smiles.json`         | SSI-DDI, SAGAN, TIGER         | chỉ khi chạy các model đó          |
| `relations_2hop.txt`     | Decagon, TIGER                | chỉ khi chạy Decagon / TIGER       |

Dataset nào thiếu file sẽ báo lỗi rõ ràng ngay lúc load (`FileNotFoundError`
kèm đường dẫn mong đợi), thay vì chạy nhầm sang data của dataset khác.

## Thêm dataset mới

1. Bỏ data vào `data/<tên>_cluster/` (+ `_random/`) và `data/initial/<tên>/`.
2. Thêm một entry vào `DATASETS` trong `../dataset_registry.py`.

Không cần sửa gì trong `trainer.py`, `data_process.py`, `utils.py` hay `models/`
— code branch theo `args.task` (`multiclass` / `multilabel`), không theo tên dataset.

## Cache

TIGER ghi cache subgraph/molecule vào chính thư mục split
(`data/<dataset>_<type>/tiger_mol_sp.json`, `data/<dataset>_<type>/rw/`),
nên cache của các dataset cũng tách bạch. Xoá thư mục cache nếu bạn thay data.
