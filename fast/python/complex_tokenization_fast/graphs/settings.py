from complex_tokenization_fast._rs import sync_settings


class _GraphSettingsMeta(type):
    _MAX_MERGE_SIZE = 2
    _ONLY_MINIMAL_MERGES = True

    @property
    def MAX_MERGE_SIZE(cls):  # noqa: N802
        return cls._MAX_MERGE_SIZE

    @MAX_MERGE_SIZE.setter
    def MAX_MERGE_SIZE(cls, value):  # noqa: N802
        cls._MAX_MERGE_SIZE = value
        sync_settings(cls._MAX_MERGE_SIZE, cls._ONLY_MINIMAL_MERGES)

    @property
    def ONLY_MINIMAL_MERGES(cls):  # noqa: N802
        return cls._ONLY_MINIMAL_MERGES

    @ONLY_MINIMAL_MERGES.setter
    def ONLY_MINIMAL_MERGES(cls, value):  # noqa: N802
        cls._ONLY_MINIMAL_MERGES = value
        sync_settings(cls._MAX_MERGE_SIZE, cls._ONLY_MINIMAL_MERGES)


class GraphSettings(metaclass=_GraphSettingsMeta):
    pass
