# Multi-Way Representation Alignment

We align latent spaces stemming from different models with a cycle-consistent procedure (GPA), a subspace intersection procedure (GCCA) and a GPA-based geometry correction procedure (GCPA) that implicitly defines a universal representational space.

---

## 🚀 Installation

First thing, install [`uv`](https://github.com/astral-sh/uv). 
The following instructions must be run from the path in which the repo was cloned, i.e. the same where this `README.md` is stored.
We are going to install `latentis` locally as we might have to edit this one.  Clone `latentis` from the following fork
```sh
    cd .. 
    git clone https://github.com/crisostomi/latentis
```
and create the virtual environment with `uv`
```sh
    cd multiway-alignment
    uv sync
    uv add --editable ../latentis/
```

## 🗂️ Datasets

Apart from the datasets present in latentis, the remaining datasets can be downloaded from here:
  - [Market-1501](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view?resourcekey=0-8nyl7K9_x37HlQm34MmrYQ)
  - Flickr8k:
    - [Text](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip)
    - [Image](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip)
    - [Audio](https://groups.csail.mit.edu/sls/downloads/flickraudio/downloads/flickr_audio.tar.gz)

---

## 📂 Project Structure

```
/
├── src/                     # Source directory
│   ├── cycloreps/  # Main package
│   │   ├── __init__.py
├── pyproject.toml           # Package configuration
├── README.md                # This file
└── LICENSE                  # License information
```

Individual scripts can be found in `src/scripts/exps/`.

---

## 👤 Maintainers


- **Donato Crisostomi**
- **Matéo Mahaut**
- **Tatiana Gaintseva**  
- **Viktor Stenby Johansson**
- **Akshit Achara**
- **Pritish Chakraborty**
- **Melih Barsbey**

---

## 📜 License

This project is licensed under the **MIT** License. See [LICENSE](LICENSE) for more details.

## 📝 Citation

If you find this code useful, please cite us:

```
@article{achara2026multi,
  title={Multi-Way Representation Alignment},
  author={Achara, Akshit and Gaintseva, Tatiana and Mahaut, Mateo and Chakraborty, Pritish and Johansson, Viktor Stenby and Barsbey, Melih and Rodol{\`a}, Emanuele and Crisostomi, Donato},
  journal={arXiv preprint arXiv:2602.06205},
  year={2026}
}
```