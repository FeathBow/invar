{-# LANGUAGE GHC2021 #-}

module Learning (prepare) where

import Invar.Learn qualified as L
import Invar.Rollout qualified as R

prepare :: L.Settings -> R.Driver scope -> R.Options -> IO (Either String (L.Plan scope))
prepare settings driver options = do
    generated <- R.run driver options
    pure $ case generated of
        Left problem -> Left (show problem)
        Right batch -> either (Left . show) Right (L.prepare settings batch)
