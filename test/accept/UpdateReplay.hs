{-# LANGUAGE GHC2021 #-}

module UpdateReplay (inspect) where

import Data.Aeson (Value)
import Data.ByteString (ByteString)
import Invar.Replay.Update qualified as Update

inspect :: (FilePath -> IO Bool) -> Update.Run -> ByteString -> IO Value
inspect directory run encoded = Update.describe <$> Update.admit directory run encoded
