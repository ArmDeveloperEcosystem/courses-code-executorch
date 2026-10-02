# Download a pinned Open Images subset for PTQ calibration.
#
# Open Images records the selected validation images as CC BY 2.0. The images
# are downloaded locally and are not distributed with this repository:
# https://storage.googleapis.com/openimages/web/factsfigures_v7.html
# https://creativecommons.org/licenses/by/2.0/

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import shutil
from urllib.request import Request, urlopen


DOWNLOAD_URL = "https://open-images-dataset.s3.amazonaws.com/validation/{image_id}.jpg"
DATASET_NAME = "open-images-v7-validation"
MAX_WORKERS = 8

# Five manually reviewed validation images from each of 20 broad categories.
# Hashes pin the exact files used to produce the course reference results.
IMAGES = [
    ("airplane", "149c3ea7a0236ac2", "cdee567b092b64e5f3bb762de2e3039ba8968b90be14c2e402b8677be06f5c85"),
    ("airplane", "7dc254d0b926bc4b", "52d39b66cebcca4e3a7fca6bd0e3ad7177c3263dd0ac84bfb2ecf97f144f4761"),
    ("airplane", "d521525f652575b8", "f6af4bfd5b186c730a734587b2bcd40698ce2c7ece03ac8003a942448aaeadfd"),
    ("airplane", "3486c144e4aa51e6", "fe5b9763971f1e0b80189af1d3321e4b804fb4f14dd286572adf51cb31aabfa6"),
    ("airplane", "40abd4f4edaa5d38", "df69aafa63515baae3071c261f85bcca0773d6238f64fab57b3e85d72814b85d"),
    ("bicycle", "626602418d76f821", "e91db255713aeda13bd01efa3cc5130f7b863695fd66c498a1389c77b732a3d5"),
    ("bicycle", "9fc6b9554a142b1a", "09f236cd81d78c3e541960c72f1593071cfa572fe39c19d86f5a9262faf2ffb7"),
    ("bicycle", "8ba474bcd50d500d", "348a2378e51b4f747ea64d8ae0f413865f3a4a8ad83907a468a071ce3668adc3"),
    ("bicycle", "46b44fb37d4150f3", "e5d98b2cddce447b5e3802fe221af037453b5b26bfe8a3f0b71bb966bfed3ca6"),
    ("bicycle", "25abef5bf1819448", "5f8a779f275b737bf9fc481709429400d32c7fe886a7d85c6afc295df61694bc"),
    ("bird", "416119d4062c5821", "f1a9fed56fee7ef868a34221f309d69dc4f94ec9d3cf4b6ea2c8c1ce2f9f76c9"),
    ("bird", "29a1c10a68c89c37", "bbf475cefdd36cf0e19628ada3689b38dbd6a6bbdad09b3897d8891bb9e6383d"),
    ("bird", "916a330ad8766ef0", "c03796646e9a925c3a44bbfb3c4434547c7338a62d361d78632d1124980c925f"),
    ("bird", "7c5a386015a1cf89", "70e7b0871e5009b3192ae47c0d4384532422c6608bc95f7c192a7a21624a074c"),
    ("bird", "5a90dff8e14eb84a", "54eca306ca28b8b9b7fc11133d4283ab3e05b6be934cd0513a4bf1e96061caf3"),
    ("boat", "c7f0fbe996ba40ff", "376e81b15fd79e8e13aeec6bbb64178e892cd2bb4c65a41fece8a85cc99703b3"),
    ("boat", "f25ecdf10954d201", "006763329e3dcef52ec7ad5ee59517975600e7c71c9669ccb86c9d2c89310ad6"),
    ("boat", "c6a98a07e6824038", "31b9e6dc3e9172c3be8686674a31a995a048562c8c13ffafb94e0c001c2a9a49"),
    ("boat", "2ff8991f4bcf7079", "900d7d4cc1f5e09e64293fff448cdca6a3c1d72553eacc0b87f51d21d5359568"),
    ("boat", "3baa86d4dfa63386", "f5c75c4a34d9707ca2074a8637461d1f5487833824dec65031c1e5dfce050779"),
    ("bottle", "dd42197939e82d8c", "be326e2da3d6ba47b0afbacad6b552d1d6bca033080b647bf44deee687ab9bcd"),
    ("bottle", "7f1c2fc5fb99412c", "2c30704ac2d991225a7f194c0b73e97e93438756f2ecf696f8aec1ce057ceeb1"),
    ("bottle", "6c6bab990f122299", "65616b5064b47d4d56b84d706dad37370a80fe8e8ace04540e5d3a031b2e0790"),
    ("bottle", "a362901a8fd28363", "19433736869be410cbc822474d14f7f1bff240ca139d35486f7156426f1f9d4c"),
    ("bottle", "4cc6516b9be3d77d", "4926810547a2360eeaad488b12b9f9d15d01d2051ec8e456814b5bbdd25a2123"),
    ("bus", "286d3c000d8d3591", "782c50ef9e5fae7224cd85d4a4dab73cc3297d1ff86dc6cc43a57554c3634562"),
    ("bus", "8c3a7ca22c0ac8d9", "eb3a2bc940e33d24ae7d6e0592d7894524f08712ef7f130c662eb58d627d292d"),
    ("bus", "ef3bccf37d3d9c80", "e6e0709e1674ff75cc773290418872f343bbb2e69f6bee5e5af87513a95e4f2d"),
    ("bus", "66889c8307d49bc6", "4905ab804e72b3de330cccd68a1fcf328da3076c7a45defa71cf610472d555c7"),
    ("bus", "295c5ba37ac90882", "f0844271ab97e143aa6547f004f01b19ca331f9965b19781f2c9b029b02b2681"),
    ("car", "001083f05db4352b", "801fcb40f503713648b3762e40e57a407bb0cb3ced33cf9715ba6fdcba20452c"),
    ("car", "081b07b5d3638c02", "4d09769e069900ccb9768f6effb36d7be25b6c0c8b4b08272b6e86c9eab90c30"),
    ("car", "2925a66accf0626d", "1443e8ea1036dedafe265565a36969d3834bded6be0402a48d835a86d76a2619"),
    ("car", "ecd716519d7f1f30", "7b701259c466bac59efd8c0721f456b455e7f82f150e91dd25184dd192d5984a"),
    ("car", "943bfd369d990229", "edcdd6d11fa40507efbf00557a3115768d3034995b47230aba47c10110a3ee4f"),
    ("cat", "0ad3f3375816f9ed", "86341b8a768616fb2fd15a82551b9495a2a3ad9bd4c34f3001e084427a7b3f2e"),
    ("cat", "d0fe01a71f83d22d", "d1e198c2f3c332a75bb98a28f1d9b865fb92cb6413183b5ad59b71c8fbb4b850"),
    ("cat", "19301fa614be04e8", "6b213dd7d436291fcacd6325bd51e383f8bdaa1a0e6dbe3a392fc3c224b856c7"),
    ("cat", "a23c8951dc4fe78a", "212d89f09cbf74a069b44836b7c5216f9f6828277750ea5769a6f7efd051a383"),
    ("cat", "b32c6db57efbee78", "a849f381c4943d9ec6df712d95da1473e8453fd72381e19b3a014fb18f2e7b32"),
    ("cattle", "971db95a917f8511", "419251e5b4b7b6b3e566d711ce6f1cb755d29ce969a1eb8faea96e589b3b6af3"),
    ("cattle", "450360d4f08b9f53", "8c9eba85cb7fddc1b30e12212285457becc74369f60654659a124e9bd117012c"),
    ("cattle", "373eb7cd1849a39d", "39c8485dd97ff37c9cd6b9f8642848aad7e95f7104d7f2cc1207c050319e5ffb"),
    ("cattle", "364bc47890f68a2c", "922f0c25355dc468a6569b62a1bd12036e00250a36e4e009cf33e84091135947"),
    ("cattle", "1f860f31115ed8bb", "635fa2dcedc6167bf0680f602d476c978f41daf2bb5bb927badb18e3f6ef4971"),
    ("chair", "842ab814cc6e2ba8", "56b2840061040d7b55ca8a8a63c49ea87425f6bf2edfd91fc65606a857e324b5"),
    ("chair", "b068ced86b534913", "eb8ed2cd733e3d90c7afa1cc1f48e13863c586b02822b21a86a05e02b7242a26"),
    ("chair", "10f64dac50ca0f16", "0913418e103b36c02d45e2639d8431a86be76aef2eeaf95fa036392f23f0cb58"),
    ("chair", "ddd78d7007d17a75", "6c6fe939e3d9ecb5162b0daebb692723b01e922af0703713de71c6fc6f653842"),
    ("chair", "715e35f77212c573", "0a7c8816e7e20f7c15e1f8b7c2fcce67aff23666d48b58ba5c9a39f0b017e18c"),
    ("couch", "bfbb778f3fa6ed9d", "9e55c8a4286f0a2970760a7e7e917cde56614060158b5ec9579f6e98655b8377"),
    ("couch", "5c24a157dc99ac6b", "bfaca8bf2f23e43fb7579f483c50a33c8d1acdb28899d56195a1c0a23281ef5d"),
    ("couch", "78767a54758dcd79", "da003b6d95285287ca3a4f432dd3ff204861cd58d67ec69cb0447198fba33a80"),
    ("couch", "9d848059d846a9f2", "26bc12a1e9b2fceffa8ef775e547f3a4efabdc2d377e8382e53038e87d47eff7"),
    ("couch", "477926ac3bcfb06e", "4b1d7d004011924c5d2d56479e660e97581c12a8bfc3defe08bf95bcdd941e14"),
    ("dog", "07074bd106e55c19", "7d96c95d167ce430bce5fe540d1498cd6dfb51a201bfb0e77750a60e508215e5"),
    ("dog", "b479cf9dcb883f6a", "27a68508a403e7243c639bfc97fff4129d091bc2e148a7d06b0224257331f7da"),
    ("dog", "65069a72ed33dea4", "d870bf015b596cb3420c9e4ccbff15216adeecee4cf601e10f8b91ded3df2d9e"),
    ("dog", "d0a67c4b313808cd", "4f59a809d1daec74552d5dec8241eb3163438d9a80173fa3a244067aac2369a4"),
    ("dog", "5c746298e5d7aef5", "2854f0a7e3fc79babe659f1f8f0edd1186995551d8541d9fb6599473a479e877"),
    ("horse", "793c395cc5cd8ae6", "2a514d8f104db6ac7222a381c314710d9366dde0d6718b08678924cc037c410c"),
    ("horse", "1175875723f994b1", "e81df8e0e57a8209634fba6a00e10ae451e684262589702137c38847cc0e6b5e"),
    ("horse", "5405ba8e009b5cee", "dbf80391c29d56c35b4c9c53a5d69bd3db800c826a19f7bc16e1bb874e229f20"),
    ("horse", "4cd65ac4eea5fbaa", "8c5e86f9ca3081c37d6742587a60049aa25c7f8334194c0e4cab72bdf7336511"),
    ("horse", "129d098f401181ed", "d5136bf7facd02f8ef9faf37d885f05cecfd7d40f2607bed0a56ca251b688010"),
    ("houseplant", "64bffc40f0d073fe", "5aebc1481aaa2a5572167c4db7471b57369f2918e94355e2bd800cdb6151a124"),
    ("houseplant", "fcfbd8e754fea64f", "ec84cb871e7996d28091f53da2c570c35fbd200c92f1efddbb6b2fa4dfd9621d"),
    ("houseplant", "8ef31f81bd70c3b9", "1bc97888434ff152842e7b7e2436dbe87b82ec668e2069109380581bf097a694"),
    ("houseplant", "060091417ccafbcf", "50a6e0541502464dd8d14e6812151c9c8f11b990705c988ef1985c77b530f1f5"),
    ("houseplant", "fe908de238143eb4", "b4f5d09526f32d04321a4499372fcafd8c773121b1c44e17325908fb218338b9"),
    ("motorcycle", "3145aa24c245d02c", "83889159d69cdba4a96622fd2226141814da6850e16df2d51df22d21713ab71e"),
    ("motorcycle", "91899fa1dd9de609", "f8da104fb9963dbf3baee290d7cd47f0185d84f956a8ec8389e4be6beb6ec289"),
    ("motorcycle", "001a995c1e25d892", "7650f085d42642e235f48bb24e6bb283042a26c264e38bb57c6159ae2d898a50"),
    ("motorcycle", "a2473495200c1a2a", "98b4bde40e6e320f12393547e568fa4bface55993f26ab91a52a57862c0126e2"),
    ("motorcycle", "64207cf6b4fdbf2f", "24f77aebf1a06d0cbdefb894fcd44e21788e8cb3cf3300dc8b9bf78e1d3d71c3"),
    ("person", "ef0d8d1a8cee60ae", "7c3617d977968650f53d3479840729f2fde17bd3718b091e336ad44d6595744b"),
    ("person", "72a7bab4f09f1194", "0100148975a98813860ca97580a4ea81dd342e0b63d51d2087a2ae2c942ce3d0"),
    ("person", "7aa0b17d541db028", "ee63e8b7044a09649e87ec3d14cb23e551d72fe4885c981d478477ed7fe21f30"),
    ("person", "8105247aae5f048c", "6e53c0428fd86704c240a76dab6c4fbeb7a4242f0043adf77f0619372f66a5e3"),
    ("person", "151ac0754da1da6d", "9a97245088acd810f4b1007b7a2353cf7452e74c72a6ac030273b10c61ea28dc"),
    ("sheep", "0d035b68ebd76f87", "f0e2fe913c6be3c9740865425a14a737181f5aba1effc422fe17fa4acac601ff"),
    ("sheep", "618de81b4b5c3cbf", "ef8b036587498b0fa90f4b319b33cb7dea8263de446e72c6abca09cad722fc7a"),
    ("sheep", "3d8ecbd83a6dd29e", "a2d21aad2018a6fc1ced6795b24492f48a03fe82fb0f32d28f664d48abe34115"),
    ("sheep", "6c7033ac862ef96f", "2604173f66d129acaf33fa0e65a46a1fba40f23200d5d9dbbcf176b2d35e2eb8"),
    ("sheep", "de046443b2e3ff24", "fa070234e392a0aee13f24a615f03f3deba7f20872d10d028acaa266ddd7006f"),
    ("table", "d95905d1f1b3ee5c", "bcc317932ef02237751098abeaef1fd4a6ab048afcf74c06951403558cd6a226"),
    ("table", "19d14cd8dc17adc0", "7a8d8522160f9566d2d17287ba5e89da81211daae05defddc796fcdd49b16566"),
    ("table", "742af27c3d962cdb", "26ed09a8e9fc1e05ba41070f582c41e0cc27067b48b6ff0addf6562695135129"),
    ("table", "f0d789f5cf4d0040", "ab5d368f03550d0a8f12edb07460bf3a5ce10b4c48c6e440d7c0f0a4e2a3698e"),
    ("table", "189a9911f39e9afd", "3230e1014e07fc4fe4a194250b0b7921b33a94a93b373c1fde553a891acb136a"),
    ("television", "11124dd4875ecf8d", "8ae4da1b2b8d129fcb3484b4ca5380abb61687b988104933d46f71ec0ed9928e"),
    ("television", "b6eb68d75fff7e92", "00fd3c9ba9aac3825c557c814aa0b9d4e383488d1d598068033ff94455d630e6"),
    ("television", "d828dedc4bb72794", "a39fb9342bdcb8e70ec71843fb761d621cae1cc3b9c53ff4be7e60563ccc7f32"),
    ("television", "4afce4e856a68aed", "db94fafc8c901fe8874fcaad8112c0bc5f5163755db5a83edc0107158fd8ef6d"),
    ("television", "84a1048891bb636c", "19b84ac6b9d2b269d90440a27a14d69e41cfe07f82931bd8816d1b01dd8d2cd8"),
    ("train", "364a5583d79ef608", "52e41414b28bff964035493a30fc090bc6d3d1159caaf0748b4a834df8c1cf67"),
    ("train", "322d9c0956597cc6", "20066926c044af56ee558b9d3b20b94cff345da3283144e9fb428088d33bdb0f"),
    ("train", "79a88682ed6b1706", "65c5d80fc10dc8b1f30d4e0e699f7a6f08b5689767eb32a9c000643051c43be0"),
    ("train", "feb3f8fd27b91d6e", "845b7dfd650d9ac7ec8ecb5fcfca85189e1fb376e410d9139ca27bd4ea6ff149"),
    ("train", "34bd6759def27335", "9e61fac02490e2ffaea0fe5f4dfe5166dba000c8a2ed239d78f12c5f0943dd13"),
]


def prepare_calibration_subset(*, data_dir: Path, out_dir: Path, count: int) -> None:
    """Download, verify, and copy a balanced subset into an ImageFolder."""
    if not 1 <= count <= len(IMAGES):
        raise ValueError(f"--count must be between 1 and {len(IMAGES)}")

    selected = select_balanced_images(count)
    cache_dir = data_dir / DATASET_NAME
    cache_dir.mkdir(parents=True, exist_ok=True)

    missing = [entry for entry in selected if not (cache_dir / f"{entry[1]}.jpg").exists()]
    if missing:
        print(f"Downloading {len(missing)} Open Images validation images to {cache_dir}...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        list(executor.map(lambda entry: download_or_verify(entry, cache_dir), selected))
    print(f"Verified {len(selected)} Open Images files.")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    for category, image_id, _sha256 in selected:
        output_class = out_dir / category
        output_class.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cache_dir / f"{image_id}.jpg", output_class / f"{image_id}.jpg")

    used_categories = len({category for category, _image_id, _sha256 in selected})
    print(f"Wrote {len(selected)} calibration images across {used_categories} categories to {out_dir}")


def select_balanced_images(count: int) -> list[tuple[str, str, str]]:
    """Select images evenly across categories while preserving the pinned order."""
    by_category: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for entry in IMAGES:
        by_category[entry[0]].append(entry)

    categories = list(by_category)
    base_count, remainder = divmod(count, len(categories))
    selected = []
    for index, category in enumerate(categories):
        category_count = base_count + (1 if index < remainder else 0)
        selected.extend(by_category[category][:category_count])
    return selected


def download_or_verify(entry: tuple[str, str, str], cache_dir: Path) -> None:
    """Download one image when absent and reject changed or incomplete files."""
    _category, image_id, expected_sha256 = entry
    path = cache_dir / f"{image_id}.jpg"
    if path.exists():
        verify_sha256(path, expected_sha256)
        return

    temporary_path = path.with_suffix(".jpg.part")
    request = Request(
        DOWNLOAD_URL.format(image_id=image_id),
        headers={"User-Agent": "ExecuTorch-on-Arm-course/1.0"},
    )
    try:
        with urlopen(request, timeout=60) as response, temporary_path.open("wb") as output:
            shutil.copyfileobj(response, output)
        verify_sha256(temporary_path, expected_sha256)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def verify_sha256(path: Path, expected: str) -> None:
    """Verify a cached or newly downloaded image."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError(
            f"SHA256 mismatch for {path}: expected {expected}, got {actual}. "
            "Delete the file and run the command again."
        )
