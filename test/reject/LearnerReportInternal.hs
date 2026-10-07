{-# LANGUAGE GHC2021 #-}
{-# LANGUAGE Safe #-}

-- Reject: it is a hidden module in the package
module LearnerReportInternal where

import Invar.Learn.Report.Internal ()
