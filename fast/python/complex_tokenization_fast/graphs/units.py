from complex_tokenization_fast._rs import (  # noqa: F401
    characters,
    clear_handlers,
    get_handlers_dict,
    register_script,
    utf8,
    utf8_clusters,
)


class _ClusterHandlersProxy:
    def clear(self):
        clear_handlers()

    def __delitem__(self, key):
        current = get_handlers_dict()
        if key not in current:
            raise KeyError(key)
        clear_handlers()
        for k, v in current.items():
            if k != key:
                register_script(k, v)

    def __contains__(self, key):
        return key in get_handlers_dict()

    def __len__(self):
        return len(get_handlers_dict())


_cluster_handlers = _ClusterHandlersProxy()
