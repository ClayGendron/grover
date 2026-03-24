"""Evidence types — backward-compatible aliases for Detail types.

All classes here are re-exported from ``detail.py``.  New code should
import from ``detail.py`` directly.  These aliases exist so that existing
imports (``from grover.models.internal.evidence import GlobEvidence``)
continue to work without changes.
"""

from grover.models.internal.detail import Detail as Evidence  # noqa: F401
from grover.models.internal.detail import GlobDetail as GlobEvidence  # noqa: F401
from grover.models.internal.detail import GraphCentralityDetail as GraphCentralityEvidence  # noqa: F401
from grover.models.internal.detail import GraphRelationshipDetail as GraphRelationshipEvidence  # noqa: F401
from grover.models.internal.detail import GrepDetail as GrepEvidence  # noqa: F401
from grover.models.internal.detail import HybridDetail as HybridEvidence  # noqa: F401
from grover.models.internal.detail import LexicalDetail as LexicalEvidence  # noqa: F401
from grover.models.internal.detail import LineMatch  # noqa: F401
from grover.models.internal.detail import ListDirDetail as ListDirEvidence  # noqa: F401
from grover.models.internal.detail import ReconcileDetail as ReconcileEvidence  # noqa: F401
from grover.models.internal.detail import ShareDetail as ShareEvidence  # noqa: F401
from grover.models.internal.detail import TrashDetail as TrashEvidence  # noqa: F401
from grover.models.internal.detail import TreeDetail as TreeEvidence  # noqa: F401
from grover.models.internal.detail import VectorDetail as VectorEvidence  # noqa: F401
from grover.models.internal.detail import VersionDetail as VersionEvidence  # noqa: F401
