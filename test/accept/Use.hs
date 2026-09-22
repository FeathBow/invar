{-# LANGUAGE GHC2021 #-}

module Use where

import Invar.Use

inspect :: BoundRun -> Either ObservationError (Maybe Rational)
inspect supplied = mean LossIncrease <$> observe supplied

established :: Claim -> Observed -> Finding
established = establish

admitted :: UseContract -> Finding -> Decision
admitted = admit
