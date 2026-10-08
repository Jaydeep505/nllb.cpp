# Supported languages

NLLB-200 covers **202 languages**, each identified by a [FLORES-200](https://github.com/facebookresearch/flores/blob/main/flores200/README.md) code of the form `<ISO 639-3>_<ISO 15924 script>`.
Pass these codes to `--src` / `--tgt` in `nllb-cli` (or the equivalent fields in the C API and server). Each code is also a special token in the model's vocabulary; the target code is forced as the first generated token.

```bash
# list the codes your GGUF actually carries
./build/bin/nllb-cli -m models/nllb-600M-f16.gguf --list-langs
```

## Common languages

| Code | Language | Script |
|---|---|---|
| `eng_Latn` | English | Latin |
| `deu_Latn` | German | Latin |
| `fra_Latn` | French | Latin |
| `spa_Latn` | Spanish | Latin |
| `ita_Latn` | Italian | Latin |
| `por_Latn` | Portuguese | Latin |
| `rus_Cyrl` | Russian | Cyrillic |
| `zho_Hans` | Chinese (Simplified) | Han (Simplified) |
| `jpn_Jpan` | Japanese | Japanese (Kanji + Kana) |
| `kor_Hang` | Korean | Hangul |
| `arb_Arab` | Arabic (Modern Standard) | Arabic |
| `hin_Deva` | Hindi | Devanagari |
| `guj_Gujr` | Gujarati | Gujarati |
| `ben_Beng` | Bengali | Bengali |
| `tur_Latn` | Turkish | Latin |
| `ukr_Cyrl` | Ukrainian | Cyrillic |

## All languages

Sorted by language name. Use your browser's find (Ctrl/Cmd+F) to search by name or code.

| Code | Language | Script |
|---|---|---|
| `ace_Arab` | Acehnese (Arabic) | Arabic |
| `ace_Latn` | Acehnese (Latin) | Latin |
| `afr_Latn` | Afrikaans | Latin |
| `aka_Latn` | Akan | Latin |
| `amh_Ethi` | Amharic | Ge'ez (Ethiopic) |
| `arb_Arab` | Arabic (Modern Standard) | Arabic |
| `hye_Armn` | Armenian | Armenian |
| `asm_Beng` | Assamese | Bengali |
| `ast_Latn` | Asturian | Latin |
| `awa_Deva` | Awadhi | Devanagari |
| `quy_Latn` | Ayacucho Quechua | Latin |
| `ban_Latn` | Balinese | Latin |
| `bam_Latn` | Bambara | Latin |
| `bjn_Arab` | Banjar (Arabic) | Arabic |
| `bjn_Latn` | Banjar (Latin) | Latin |
| `bak_Cyrl` | Bashkir | Cyrillic |
| `eus_Latn` | Basque | Latin |
| `bel_Cyrl` | Belarusian | Cyrillic |
| `bem_Latn` | Bemba | Latin |
| `ben_Beng` | Bengali | Bengali |
| `bho_Deva` | Bhojpuri | Devanagari |
| `bos_Latn` | Bosnian | Latin |
| `bug_Latn` | Buginese | Latin |
| `bul_Cyrl` | Bulgarian | Cyrillic |
| `mya_Mymr` | Burmese | Myanmar |
| `yue_Hant` | Cantonese | Han (Traditional) |
| `cat_Latn` | Catalan | Latin |
| `ceb_Latn` | Cebuano | Latin |
| `tzm_Tfng` | Central Atlas Tamazight | Tifinagh |
| `ayr_Latn` | Central Aymara | Latin |
| `knc_Arab` | Central Kanuri (Arabic) | Arabic |
| `knc_Latn` | Central Kanuri (Latin) | Latin |
| `ckb_Arab` | Central Kurdish | Arabic |
| `hne_Deva` | Chhattisgarhi | Devanagari |
| `zho_Hans` | Chinese (Simplified) | Han (Simplified) |
| `zho_Hant` | Chinese (Traditional) | Han (Traditional) |
| `cjk_Latn` | Chokwe | Latin |
| `crh_Latn` | Crimean Tatar | Latin |
| `hrv_Latn` | Croatian | Latin |
| `ces_Latn` | Czech | Latin |
| `dan_Latn` | Danish | Latin |
| `prs_Arab` | Dari | Arabic |
| `nld_Latn` | Dutch | Latin |
| `dyu_Latn` | Dyula | Latin |
| `dzo_Tibt` | Dzongkha | Tibetan |
| `pan_Guru` | Eastern Panjabi | Gurmukhi |
| `ydd_Hebr` | Eastern Yiddish | Hebrew |
| `arz_Arab` | Egyptian Arabic | Arabic |
| `eng_Latn` | English | Latin |
| `epo_Latn` | Esperanto | Latin |
| `est_Latn` | Estonian | Latin |
| `ewe_Latn` | Ewe | Latin |
| `fao_Latn` | Faroese | Latin |
| `fij_Latn` | Fijian | Latin |
| `fin_Latn` | Finnish | Latin |
| `fon_Latn` | Fon | Latin |
| `fra_Latn` | French | Latin |
| `fur_Latn` | Friulian | Latin |
| `glg_Latn` | Galician | Latin |
| `lug_Latn` | Ganda | Latin |
| `kat_Geor` | Georgian | Georgian |
| `deu_Latn` | German | Latin |
| `ell_Grek` | Greek | Greek |
| `grn_Latn` | Guarani | Latin |
| `guj_Gujr` | Gujarati | Gujarati |
| `hat_Latn` | Haitian Creole | Latin |
| `khk_Cyrl` | Halh Mongolian | Cyrillic |
| `hau_Latn` | Hausa | Latin |
| `heb_Hebr` | Hebrew | Hebrew |
| `hin_Deva` | Hindi | Devanagari |
| `hun_Latn` | Hungarian | Latin |
| `isl_Latn` | Icelandic | Latin |
| `ibo_Latn` | Igbo | Latin |
| `ilo_Latn` | Ilocano | Latin |
| `ind_Latn` | Indonesian | Latin |
| `gle_Latn` | Irish | Latin |
| `ita_Latn` | Italian | Latin |
| `jpn_Jpan` | Japanese | Japanese (Kanji + Kana) |
| `jav_Latn` | Javanese | Latin |
| `kac_Latn` | Jingpho | Latin |
| `kbp_Latn` | Kabiyè | Latin |
| `kea_Latn` | Kabuverdianu | Latin |
| `kab_Latn` | Kabyle | Latin |
| `kam_Latn` | Kamba | Latin |
| `kan_Knda` | Kannada | Kannada |
| `kas_Arab` | Kashmiri (Arabic) | Arabic |
| `kas_Deva` | Kashmiri (Devanagari) | Devanagari |
| `kaz_Cyrl` | Kazakh | Cyrillic |
| `khm_Khmr` | Khmer | Khmer |
| `kon_Latn` | Kikongo | Latin |
| `kik_Latn` | Kikuyu | Latin |
| `kmb_Latn` | Kimbundu | Latin |
| `kin_Latn` | Kinyarwanda | Latin |
| `kor_Hang` | Korean | Hangul |
| `kir_Cyrl` | Kyrgyz | Cyrillic |
| `lao_Laoo` | Lao | Lao |
| `ltg_Latn` | Latgalian | Latin |
| `lvs_Latn` | Latvian | Latin |
| `lij_Latn` | Ligurian | Latin |
| `lim_Latn` | Limburgish | Latin |
| `lin_Latn` | Lingala | Latin |
| `lit_Latn` | Lithuanian | Latin |
| `lmo_Latn` | Lombard | Latin |
| `lua_Latn` | Luba-Kasai | Latin |
| `luo_Latn` | Luo | Latin |
| `ltz_Latn` | Luxembourgish | Latin |
| `mkd_Cyrl` | Macedonian | Cyrillic |
| `mag_Deva` | Magahi | Devanagari |
| `mai_Deva` | Maithili | Devanagari |
| `mal_Mlym` | Malayalam | Malayalam |
| `mlt_Latn` | Maltese | Latin |
| `mri_Latn` | Maori | Latin |
| `mar_Deva` | Marathi | Devanagari |
| `mni_Beng` | Meitei (Bengali) | Bengali |
| `acm_Arab` | Mesopotamian Arabic | Arabic |
| `min_Latn` | Minangkabau | Latin |
| `lus_Latn` | Mizo | Latin |
| `ary_Arab` | Moroccan Arabic | Arabic |
| `mos_Latn` | Mossi | Latin |
| `ars_Arab` | Najdi Arabic | Arabic |
| `npi_Deva` | Nepali | Devanagari |
| `fuv_Latn` | Nigerian Fulfulde | Latin |
| `azj_Latn` | North Azerbaijani | Latin |
| `apc_Arab` | North Levantine Arabic | Arabic |
| `kmr_Latn` | Northern Kurdish | Latin |
| `nso_Latn` | Northern Sotho | Latin |
| `uzn_Latn` | Northern Uzbek | Latin |
| `nob_Latn` | Norwegian Bokmål | Latin |
| `nno_Latn` | Norwegian Nynorsk | Latin |
| `nus_Latn` | Nuer | Latin |
| `nya_Latn` | Nyanja | Latin |
| `oci_Latn` | Occitan | Latin |
| `ory_Orya` | Odia | Odia |
| `pag_Latn` | Pangasinan | Latin |
| `pap_Latn` | Papiamento | Latin |
| `plt_Latn` | Plateau Malagasy | Latin |
| `pol_Latn` | Polish | Latin |
| `por_Latn` | Portuguese | Latin |
| `ron_Latn` | Romanian | Latin |
| `run_Latn` | Rundi | Latin |
| `rus_Cyrl` | Russian | Cyrillic |
| `smo_Latn` | Samoan | Latin |
| `sag_Latn` | Sango | Latin |
| `san_Deva` | Sanskrit | Devanagari |
| `sat_Beng` | Santali | Bengali |
| `srd_Latn` | Sardinian | Latin |
| `gla_Latn` | Scottish Gaelic | Latin |
| `srp_Cyrl` | Serbian | Cyrillic |
| `shn_Mymr` | Shan | Myanmar |
| `sna_Latn` | Shona | Latin |
| `scn_Latn` | Sicilian | Latin |
| `szl_Latn` | Silesian | Latin |
| `snd_Arab` | Sindhi | Arabic |
| `sin_Sinh` | Sinhala | Sinhala |
| `slk_Latn` | Slovak | Latin |
| `slv_Latn` | Slovenian | Latin |
| `som_Latn` | Somali | Latin |
| `azb_Arab` | South Azerbaijani | Arabic |
| `ajp_Arab` | South Levantine Arabic | Arabic |
| `pbt_Arab` | Southern Pashto | Arabic |
| `sot_Latn` | Southern Sotho | Latin |
| `dik_Latn` | Southwestern Dinka | Latin |
| `spa_Latn` | Spanish | Latin |
| `zsm_Latn` | Standard Malay | Latin |
| `bod_Tibt` | Standard Tibetan | Tibetan |
| `sun_Latn` | Sundanese | Latin |
| `swh_Latn` | Swahili | Latin |
| `ssw_Latn` | Swati | Latin |
| `swe_Latn` | Swedish | Latin |
| `acq_Arab` | Ta'izzi-Adeni Arabic | Arabic |
| `tgl_Latn` | Tagalog | Latin |
| `tgk_Cyrl` | Tajik | Cyrillic |
| `taq_Latn` | Tamasheq (Latin) | Latin |
| `taq_Tfng` | Tamasheq (Tifinagh) | Tifinagh |
| `tam_Taml` | Tamil | Tamil |
| `tat_Cyrl` | Tatar | Cyrillic |
| `tel_Telu` | Telugu | Telugu |
| `tha_Thai` | Thai | Thai |
| `tir_Ethi` | Tigrinya | Ge'ez (Ethiopic) |
| `tpi_Latn` | Tok Pisin | Latin |
| `als_Latn` | Tosk Albanian | Latin |
| `tso_Latn` | Tsonga | Latin |
| `tsn_Latn` | Tswana | Latin |
| `tum_Latn` | Tumbuka | Latin |
| `aeb_Arab` | Tunisian Arabic | Arabic |
| `tur_Latn` | Turkish | Latin |
| `tuk_Latn` | Turkmen | Latin |
| `twi_Latn` | Twi | Latin |
| `ukr_Cyrl` | Ukrainian | Cyrillic |
| `umb_Latn` | Umbundu | Latin |
| `urd_Arab` | Urdu | Arabic |
| `uig_Arab` | Uyghur | Arabic |
| `vec_Latn` | Venetian | Latin |
| `vie_Latn` | Vietnamese | Latin |
| `war_Latn` | Waray | Latin |
| `cym_Latn` | Welsh | Latin |
| `gaz_Latn` | West Central Oromo | Latin |
| `pes_Arab` | Western Persian | Arabic |
| `wol_Latn` | Wolof | Latin |
| `xho_Latn` | Xhosa | Latin |
| `yor_Latn` | Yoruba | Latin |
| `zul_Latn` | Zulu | Latin |
