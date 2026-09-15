module PolicyRevision where

import qualified Invar.Policy as Policy

-- Reject: Not in scope: record field
unchecked :: Policy.Description -> Policy.Description
unchecked selected = selected {Policy.revision = ""}
