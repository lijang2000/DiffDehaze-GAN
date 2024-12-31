

_MODELS = {}
def register_model(cls=None, *, name=None):
      """A decorator for registering model classes."""
      def _register(cls):
            if name is None:
                local_name = cls.__name__
            else:
                local_name = name
            if local_name in _MODELS:
                raise ValueError(f'Already registered model with name: {local_name}')
            _MODELS[local_name] = cls
            return cls

      if cls is None:
            return _register
      else:
            return _register(cls)