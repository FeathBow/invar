module History (Pair (..), genPair, cacheLowered, cachePreserved) where

import Data.List.NonEmpty (NonEmpty (..))
import Hedgehog (Gen)
import Hedgehog.Gen qualified as Gen
import Hedgehog.Range qualified as Range
import Invar.Spec.Request
import Model

data Pair = Pair
    { options :: Settings
    , initial :: State
    , leftHistory :: [Event]
    , rightHistory :: [Event]
    }
    deriving (Eq, Show)

maxPrompt, maxTokens, maxSeed, maxStride :: Int
maxPrompt = 4
maxTokens = 5
maxSeed = 15
maxStride = 3

genSettings :: Gen Settings
genSettings =
    Settings
        <$> Gen.int (Range.constant 0 (2 ^ tableWidth - 1))
        <*> Gen.bool
        <*> (fromIntegral <$> Gen.int (Range.constant 1 maxStride))

genCore :: Gen (Core Int Bool)
genCore = do
    seed <- Gen.int (Range.constant 0 maxSeed)
    first <- Gen.bool
    rest <- Gen.list (Range.linear 0 (maxPrompt - 1)) Gen.bool
    cap <- fromIntegral <$> Gen.int (Range.constant 1 maxTokens)
    pure (require (mkCore seed (first :| rest) cap))

genPair :: Gen Pair
genPair = do
    settings <- genSettings
    targetCore <- genCore
    otherCore <- genCore
    cancelled <- Gen.bool
    let pool = require (initialize [(target, targetCore), (other, otherCore)])
        baseline = require (completion settings pool target)
        otherScript =
            if cancelled
                then [Event other Admit, Event other (Preempt 0), Event other Cancel]
                else require (completion settings pool other)
    perturbed <- perturb settings pool baseline
    mixed <- case otherScript of
        [] -> error "Other-request script must include admission"
        first : rest -> (first :) <$> interleave perturbed rest
    pure (Pair settings pool (baseline ++ otherScript) mixed)

interleave :: [value] -> [value] -> Gen [value]
interleave [] ys = pure ys
interleave xs [] = pure xs
interleave (x : xs) (y : ys) = do
    chooseLeft <- Gen.bool
    if chooseLeft
        then (x :) <$> interleave xs (y : ys)
        else (y :) <$> interleave (x : xs) ys

perturb :: Settings -> State -> [Event] -> Gen [Event]
perturb settings = go
  where
    transition = step (semantics settings)
    go _ [] = pure []
    go config (event : rest) = do
        injected <-
            if action event == Decode
                then interruption settings config (requestId event)
                else pure []
        let script = injected ++ [event]
            updated = last (require (execute transition config script))
        (script ++) <$> go updated rest

interruption :: Settings -> State -> RequestId -> Gen [Event]
interruption settings config rid = do
    let position = cached (requireLookup rid config)
    preserve <- Gen.bool
    retained <- if preserve then pure position else Gen.integral (Range.constant 0 position)
    let prefix = [Event rid (Preempt retained), Event rid Resume]
        resumed = last (require (execute (step (semantics settings)) config prefix))
    pure (prefix ++ refill settings resumed rid)

refill :: Settings -> State -> RequestId -> [Event]
refill settings config rid
    | cached request == extent request = []
    | cached request > extent request = error "Cannot refill a cache-ahead fixture"
    | otherwise = case step (semantics settings) config event of
        Rejected reason -> error (show reason)
        Applied updated -> event : refill settings updated rid
  where
    request = requireLookup rid config
    event = Event rid Chunk

cacheLowered :: [State] -> [Event] -> Bool
cacheLowered states events = or (zipWith lowers states events)
  where
    lowers config (Event rid (Preempt position)) = position < cached (requireLookup rid config)
    lowers _ _ = False

cachePreserved :: [State] -> [Event] -> Bool
cachePreserved states events = or (zipWith preserves states events)
  where
    preserves config (Event rid (Preempt position)) = position > 0 && position == cached (requireLookup rid config)
    preserves _ _ = False
