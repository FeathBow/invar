{-# LANGUAGE GHC2021 #-}

module Histories (inspect, initial, pair) where

import Control.Monad (void)
import Data.ByteString (ByteString)
import Invar.History qualified as History
import Invar.History.Initial qualified as Initial
import Invar.Learn qualified as Learn
import Invar.Learn.State (Decoder)

inspect :: Decoder -> History.Declaration -> (ByteString, ByteString) -> IO History.Checked
inspect = History.admit

initial :: Decoder -> (Learn.Settings, FilePath, Initial.Random) -> Initial.Source -> IO Initial.Checked
initial = Initial.admit

pair :: (History.Checked, History.Checked) -> IO ()
pair = void . History.compare
