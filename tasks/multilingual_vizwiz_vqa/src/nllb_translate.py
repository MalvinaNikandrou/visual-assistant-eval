from tqdm import tqdm

import torch
from transformers import pipeline
import sacrebleu

from constants import Language, lang_flores200_codes, SEED

torch.manual_seed(SEED)


class DatasetTranslation:
    """Translate a dataset.

    dataset: Dataset. Any iterator of strings to be translated
    (e.g. a list of strings or a pytorch dataset where the __getitem__ returns the string to be translated)
    """

    def __init__(
        self,
        model_name: str,
        batch_size: int = 1,
        device: str = "cuda",
        **kwargs,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.translation_kwargs = kwargs

    def __call__(self, dataset, src_lang: Language, tgt_lang: Language):
        print(f"Loading the {tgt_lang.name} translation pipeline")
        src_lang_flores_code = lang_flores200_codes[src_lang]
        tgt_lang_flores_code = lang_flores200_codes[tgt_lang]
        translator = pipeline(
            "translation",
            model=self.model_name,
            device=self.device,
            src_lang=src_lang_flores_code,
            tgt_lang=tgt_lang_flores_code,
            torch_dtype=torch.half,
        )
        print(f"Translating to {tgt_lang.name}")
        translations = []
        
        total = len(dataset)
        desc = f"Translating..."
        for translation in tqdm(
            translator(
                dataset,
                batch_size=self.batch_size,
                do_sample=self.translation_kwargs.get("do_sample", False),
                num_return_sequences=self.translation_kwargs.get("num_return_sequences", 1),
                top_p=self.translation_kwargs.get("top_p"),
                top_k=self.translation_kwargs.get("top_k"),
                temperature=self.translation_kwargs.get("temperature"),
                ),
            total=total,
            desc=desc,
        ):
            if isinstance(translation, list):
                continue
            translations.append(translation["translation_text"])
        assert len(translations) == total, f"Got only {len(translations)} from {total} data points"
        return translations


class NLLBTranslate:
    def __init__(self, n_samples: int, model_name: str, batch_size: int = 1, device: str = "cuda", **kwargs):
        self.n_samples = n_samples
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.translation_kwargs = {"do_sample": True, "num_return_sequences": 1, "top_k": 50, "top_p": 0.95, "temperature": 0.85}
        
    def __call__(self, dataset: list[str], src_lang: Language, tgt_lang: Language):
        # Get N translations for each sentence in the dataset with sampling
        translator = DatasetTranslation(
                model_name=self.model_name,
                batch_size=self.batch_size,
                device=self.device,
                translation_kwargs=self.translation_kwargs
                
            )
        translation_samples = []
        for _ in range(self.n_samples):
            translation = translator(dataset, src_lang, tgt_lang)
            translation_samples.append(translation)
        
        # Get a back translation for each translation (no sampling)
        back_translation_samples = []
        translator = DatasetTranslation(
                model_name=self.model_name,
                batch_size=self.batch_size,
                device=self.device,
            )
        for translation in translation_samples:
            back_translation = translator(translation, tgt_lang, src_lang)
            back_translation_samples.append(back_translation)
        
        # Select the best translation based on the back translation BLEU score with the original sentence
        best_translations = []
        
        for i, original in enumerate(dataset):
            translations = [sample[i] for sample in translation_samples]
            back_translations = [sample[i] for sample in back_translation_samples]
            score = 0
            idx = 0
            best_index = 0
            for translation, back_translation in zip(translations, back_translations):
                sscore = sacrebleu.compat.sentence_bleu(
                    back_translation, [original],
                    lowercase=True,
                    use_effective_order=True
                ).score
                if sscore > score:
                    score = sscore
                    best_index = idx
                idx += 1
            best_translations.append(translations[best_index])
        return best_translations