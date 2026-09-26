{-# LANGUAGE OverloadedStrings #-}

module UsePrepare (usePrepare) where

import Data.ByteString.Char8 qualified as Bytes
import Data.Either (isLeft)
import Data.List (nub, sort)
import Hedgehog
import Invar.Numerical qualified as N
import Invar.Use qualified as U
import Invar.Use.Prepare qualified as Prepare
import UseAdmission (contractFor)
import UseBernstein (contract, seedTotal, unchanged, unitTotal)
import UseFixture qualified as F

usePrepare :: Group
usePrepare =
    Group
        "Contract preparation"
        [ ("predicted premises equal those an admission relies on", once predicted)
        , ("the basis digest follows the fixed serialization", once basis)
        , ("units follow the domain's aggregation, not the input count", once counted)
        ]
  where
    once = withTests 1 . property

relied :: U.UseContract -> U.BoundRun -> PropertyT IO [U.Premise]
relied requested supplied = do
    observed <- evalEither (U.observe supplied)
    case U.admit requested (U.establish (U.Required (U.scope observed) (U.criterion requested)) observed) of
        U.Admitted result -> pure (sort (nub (map (U.premise . U.supporting) (U.conditions result))))
        other -> annotateShow other >> failure

predicted :: PropertyT IO ()
predicted = do
    finite <- F.fixture F.trials
    finiteContract <- contractFor finite
    relied finiteContract finite >>= (=== Prepare.expectedPremises finiteContract)
    repeated <- F.repeated 2 F.trials
    base <- contractFor repeated
    let invariant = base {U.criterion = (U.criterion base) {U.invarianceRequirements = [U.InvarianceRequirement N.SameBehaviorBits 3 "Fixture invariance only"]}}
    relied invariant repeated >>= (=== Prepare.expectedPremises invariant)
    population <- F.fixture (unchanged unitTotal)
    bernstein <- contract population
    relied bernstein population >>= (=== Prepare.expectedPremises bernstein)

basis :: PropertyT IO ()
basis = do
    let files = [("isolation/log.txt", "229fbbd5cefa5a379910e6719e9eaa46ca7f3a58a54f7c32a6ab8d80ff331569"), ("freeze/record.md", "a9621fd401cf5a3fb3a314be62455fc19a1f0a478381a0c03562485f98c0e207")]
    serialized <- evalEither (Prepare.basisBytes files)
    serialized === Bytes.pack "[[\"freeze/record.md\",\"a9621fd401cf5a3fb3a314be62455fc19a1f0a478381a0c03562485f98c0e207\"],[\"isolation/log.txt\",\"229fbbd5cefa5a379910e6719e9eaa46ca7f3a58a54f7c32a6ab8d80ff331569\"]]"
    Prepare.basisBytes [("a\"b\nc", "d")] === Right (Bytes.pack "[[\"a\\\"b\\nc\",\"d\"]]")
    assert (all (isLeft . Prepare.basisBytes) [[], [("/abs/file", "x")], [("a/../b", "x")], [("./a", "x")], [("a", "x"), ("a", "y")], [("a\\b", "x")]])

counted :: PropertyT IO ()
counted = do
    population <- F.fixture (unchanged unitTotal)
    bernstein <- contract population
    let report = Prepare.units bernstein
    Prepare.inputCount report === fromIntegral (unitTotal * seedTotal)
    Prepare.unitCount report === fromIntegral unitTotal
    Prepare.seedsPerUnit report === [(fromIntegral seedTotal, fromIntegral unitTotal)]
    Prepare.candidateExecutions report === 1
