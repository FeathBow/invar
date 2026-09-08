{-# LANGUAGE GHC2021 #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE Safe #-}

module Construction (sumItems, unused, nested) where

import Invar.Construct qualified as C
import Invar.Literal qualified as L

sumItems :: C.Statement
sumItems = C.emit @"out" @"sum" @'[ 'C.Semantic "items"] (C.foldSequence (C.sequenceSource @('C.Semantic "items") @Rational) (C.number @0 @1) (C.add (C.variable C.Here) (C.variable (C.There C.Here))))

unused :: C.Statement
unused = C.emit @"out" @"value" @'[] (C.letValue (C.booleanSource @('C.Operational "history")) C.true)

nested :: C.Statement
nested = C.emit @"out" @"value" @'[] (C.literal (L.sequence [L.record (L.field @"flags" (L.sequence [L.true, L.false]) L.emptyFields)]))
