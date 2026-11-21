"""Error types. Deliberately few; each one maps to a gate failure rather than a warning."""


class VeraError(Exception):
    """Base error for VERA."""


class ContractError(VeraError):
    """A data contract is malformed or a hash-relevant value is not canonicalizable."""


class AuthorityError(VeraError):
    """An action was attempted without valid authority."""


class RoleViolation(VeraError):
    """A database identity attempted an operation outside its role."""


class VerifierError(VeraError):
    """The verifier could not produce a trustworthy verdict."""


class ParserError(VeraError):
    """A model-written weight payload was rejected."""


class ManifestError(VeraError):
    """A manifest is invalid, duplicated, or violates the frozen population."""


class HoldoutSealError(VeraError):
    """The sealed holdout set is missing, misordered, or authored by the wrong party."""
