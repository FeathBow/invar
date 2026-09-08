{-# LANGUAGE GHC2021 #-}

module Loop (execute) where

import Invar.Loop qualified as L

execute :: L.Config -> L.Cycle -> IO (Either L.Error (Either L.Error L.Checkpoint))
execute config workload = L.withDriver config $ \driver -> fmap L.current <$> L.run driver workload
