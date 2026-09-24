{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}

module UseBernstein (useBernstein) where

import Calls (change, field)
import Control.Monad (forM_)
import Data.Aeson (Value (..), encode)
import Data.ByteString.Lazy qualified as Lazy
import Data.Either (isLeft)
import Data.Foldable (toList)
import Data.List (sort)
import Data.Ratio ((%))
import Hedgehog
import Invar.Spec.Evidence qualified as E
import Invar.Use qualified as U
import UseAdmission (contractFor)
import UseFixture qualified as F

useBernstein :: Group
useBernstein =
    Group
        "Empirical Bernstein use admission"
        [ ("32 prompts with four seeds retain the zero-variance correction", once zeroVariance)
        , ("one prompt with four seeds cannot supply a sample variance", once singleUnit)
        , ("variance follows unit means under unequal replication of outcomes", once groupedVariance)
        , ("an uninformative upper bound stays unknown rather than refuting the candidate", once insufficient)
        , ("joint admission retains the distinct Bernstein method and premises", once admission)
        , ("old sampling reliance cannot replace any Bernstein-specific premise", once reliance)
        , ("finite and Hoeffding findings cannot fulfill Bernstein requirements", once strength)
        , ("codec method tags select distinct requirements and findings", once encoding)
        , ("unsupported metrics and changed scopes remain unknown", once refusals)
        ]
  where
    once = withTests 1 . property

population :: U.Population
population = U.Population "Fixture population only" "Independent identically distributed prompt draws are an explicit premise" "Declared distinct seeds within each prompt; fixture only" (1 % 20) (1 % 20) (1 % 10)

unitTotal, seedTotal :: Int
unitTotal = 32
seedTotal = 4

trialsFor :: Int -> (Int -> Integer -> Bool) -> [F.Trial]
trialsFor count incorrect =
    [ F.Trial (show question ++ "/" ++ show seed) ("question " ++ show question) seed "#### 12" "#### 12" (if incorrect question seed then "#### 0" else "#### 12") False
    | question <- [1 .. count]
    , seed <- [1 .. fromIntegral seedTotal]
    ]

unchanged :: Int -> [F.Trial]
unchanged count = trialsFor count (\_ _ -> False)

contract :: U.BoundRun -> PropertyT IO U.UseContract
contract supplied = do
    base <- contractFor supplied
    let requested = U.LossRequirement (U.Budget (1 % 3) "Wide reference fixture boundary") (U.Budget (3 % 5) "Wide regression fixture boundary") (U.EmpiricalBernsteinPopulation population)
    pure base {U.criterion = (U.criterion base) {U.lossRequirement = Just requested}}

withStandard :: U.Standard -> U.UseContract -> U.UseContract
withStandard selected base = base {U.criterion = criterion {U.lossRequirement = replace <$> U.lossRequirement criterion}}
  where
    criterion = U.criterion base
    replace requested = requested {U.standard = selected}

established :: U.UseContract -> U.Observed -> U.Finding
established requested observed = U.establish (U.Required (U.scope observed) (U.criterion requested)) observed

claim :: U.Observed -> U.Metric -> Rational -> U.Claim
claim observed = U.EmpiricalBernsteinClaim (U.scope observed) population

boundFor :: U.Observed -> U.Metric -> PropertyT IO U.Confidence
boundFor observed metric = evalMaybe (U.confidence (claim observed metric 1) observed)

accepted :: U.Decision -> PropertyT IO U.Admission
accepted (U.Admitted value) = pure value
accepted other = annotateShow other >> failure

unknown :: U.Decision -> PropertyT IO [U.AdmissionProblem]
unknown (U.Undetermined problems) = pure problems
unknown other = annotateShow other >> failure

zeroVariance :: PropertyT IO ()
zeroVariance = do
    supplied <- F.fixture (unchanged unitTotal)
    observed <- evalEither (U.observe supplied)
    length (U.cases supplied) === unitTotal * seedTotal
    U.mean U.LossIncrease observed === Just 0
    regression <- boundFor observed U.LossIncrease
    reference <- boundFor observed U.ReferenceLoss
    U.unitCount regression === fromIntegral unitTotal
    U.alpha regression === 1 % 20
    assert (U.width regression > 555315186 % 1000000000 && U.width regression < 555315187 % 1000000000)
    U.width regression === 2 * U.width reference
    U.upper regression === U.width regression
    U.upper reference === U.width reference

singleUnit :: PropertyT IO ()
singleUnit = do
    supplied <- F.fixture (unchanged 1)
    observed <- evalEither (U.observe supplied)
    length (U.cases supplied) === seedTotal
    forM_ [U.ReferenceLoss, U.LossIncrease] $ \metric -> do
        let target = claim observed metric 1
        U.confidence target observed === Nothing
        U.finding (U.establish target observed) === E.Unknown (E.TaskLossProblem (U.InsufficientUnits 1 2))

groupedVariance :: PropertyT IO ()
groupedVariance = do
    let values = trialsFor unitTotal (\question seed -> question > unitTotal `div` 2 && seed <= 2)
        expanded =
            [ value {F.name = F.name value ++ "/copy/" ++ show copy, F.seed = F.seed value + fromIntegral (seedTotal * (copy - 1))}
            | (index, value) <- zip [0 :: Int ..] values
            , copy <- [1 .. 1 + (index `div` seedTotal) `mod` 3]
            ]
    supplied <- F.fixture values
    repeated <- F.fixture expanded
    observed <- evalEither (U.observe supplied)
    replicated <- evalEither (U.observe repeated)
    let losses = sort . map U.candidateLoss . toList . U.units
        expected = replicate (unitTotal `div` 2) (Just 0) ++ replicate (unitTotal `div` 2) (Just (1 % 2))
    losses observed === expected
    losses replicated === expected
    U.mean U.LossIncrease observed === Just (1 % 4)
    assert (length expanded > length values)
    assert (any ((/= seedTotal) . length . U.members) (U.units replicated))
    original <- boundFor observed U.LossIncrease
    duplicated <- boundFor replicated U.LossIncrease
    U.unitCount original === fromIntegral unitTotal
    duplicated === original
    assert (U.width original > 677276303 % 1000000000 && U.width original < 677276304 % 1000000000)
    U.upper original === 1 % 4 + U.width original

insufficient :: PropertyT IO ()
insufficient = do
    supplied <- F.fixture (unchanged unitTotal)
    observed <- evalEither (U.observe supplied)
    base <- contract supplied
    bound <- boundFor observed U.LossIncrease
    let budget = 1 % 50
        problem = U.InsufficientLossBound U.LossIncrease budget bound
        selected = (U.criterion base) {U.lossRequirement = fmap (\requirement -> requirement {U.regressionCeiling = U.Budget budget "Deliberately insufficient fixture boundary"}) (U.lossRequirement (U.criterion base))}
        requested = base {U.criterion = selected}
    U.finding (U.establish (claim observed U.LossIncrease budget) observed) === E.Unknown (E.TaskLossProblem problem)
    reasons <- unknown (U.admit requested (established requested observed))
    reasons === [U.FindingUnknown (E.TaskLossProblem problem)]

samplingPremises :: [U.Premise]
samplingPremises = [U.BernsteinIndependentUnits, U.BernsteinIdenticalUnits, U.BernsteinReplicateSamplingLaw]

admission :: PropertyT IO ()
admission = do
    supplied <- F.fixture (unchanged unitTotal)
    observed <- evalEither (U.observe supplied)
    requested <- contract supplied
    result <- accepted (U.admit requested (established requested observed))
    U.admissionContract result === requested
    U.admissionScope result === U.scope observed
    let methods = E.methods (U.evidence result)
        used = map (U.premise . U.supporting) (U.conditions result)
    [metric | E.EmpiricalBernsteinBound metric _ <- methods] === [U.ReferenceLoss, U.LossIncrease]
    [() | E.HoeffdingBound {} <- methods] === []
    U.bounds (established requested observed) === [(metric, bound) | E.EmpiricalBernsteinBound metric bound <- methods]
    forM_ (U.lossClaims (U.scope observed) (U.criterion requested)) $ \selected -> case selected of
        U.EmpiricalBernsteinClaim _ _ metric _ -> lookup metric (U.bounds (established requested observed)) === U.confidence selected observed
        other -> annotateShow other >> failure
    forM_ samplingPremises $ \premise -> assert (premise `elem` used)
    assert (U.IndependentUnits `notElem` used && U.ReplicateSamplingLaw `notElem` used)

reliance :: PropertyT IO ()
reliance = do
    supplied <- F.fixture (unchanged unitTotal)
    observed <- evalEither (U.observe supplied)
    base <- contract supplied
    forM_ samplingPremises $ \missing -> do
        let requested = base {U.reliance = filter ((/= missing) . U.premise) (U.reliance base)}
        reasons <- unknown (U.admit requested (established requested observed))
        assert (not (null reasons))
        forM_ reasons $ \case
            U.MissingReliance obligation -> E.specification obligation === "paired-loss-bernstein-mp2009/v1"
            other -> annotateShow other >> failure

strength :: PropertyT IO ()
strength = do
    supplied <- F.fixture (unchanged unitTotal)
    observed <- evalEither (U.observe supplied)
    requested <- contract supplied
    forM_ [U.FiniteDomain, U.HoeffdingPopulation population] $ \standard -> do
        let other = withStandard standard requested
        reasons <- unknown (U.admit requested (established other observed))
        assert (U.RequirementsMismatch `elem` reasons)
        reverseReasons <- unknown (U.admit other (established requested observed))
        assert (U.RequirementsMismatch `elem` reverseReasons)

encoding :: PropertyT IO ()
encoding = do
    supplied <- F.fixture (unchanged unitTotal)
    observed <- evalEither (U.observe supplied)
    requested <- contract supplied
    let encoded = U.describeContract requested
        criterion = field "criterion" encoded
        loss = field "loss" criterion
        standard = field "standard" loss
        bytes = Lazy.toStrict . encode
        tagged tag = change "criterion" (change "loss" (change "standard" (change "kind" (String tag) standard) loss) criterion) encoded
    field "kind" standard === String "population_bernstein_mp2009"
    U.decodeContract (bytes encoded) === Right requested
    other <- evalEither (U.decodeContract (bytes (tagged "population_hoeffding")))
    other === withStandard (U.HoeffdingPopulation population) requested
    reasons <- unknown (U.admit other (established requested observed))
    assert (U.RequirementsMismatch `elem` reasons)
    assert (isLeft (U.decodeContract (bytes (tagged "population_unknown"))))
    let bernstein = U.establish (claim observed U.LossIncrease 1) observed
        hoeffding = U.establish (U.PopulationClaim (U.scope observed) population U.LossIncrease 1) observed
    field "strength" (U.describeFinding bernstein) === String "population_bernstein_mp2009"
    field "strength" (U.describeFinding hoeffding) === String "population_hoeffding"

refusals :: PropertyT IO ()
refusals = do
    supplied <- F.fixture (unchanged 2)
    observed <- evalEither (U.observe supplied)
    let unsupported = claim observed U.CandidateLoss 1
        changedDomain = (U.domain supplied) {U.unitDefinition = "Different declared sampling unit meaning"}
    U.confidence unsupported observed === Nothing
    U.finding (U.establish unsupported observed) === E.Unknown (E.TaskLossProblem (U.UnsupportedPopulationMetric U.CandidateLoss))
    other <- evalEither (U.observe supplied {U.domain = changedDomain})
    let target = claim observed U.LossIncrease 1
        mismatch = U.ScopeMismatch (U.scopeId (U.scope observed)) (U.scopeId (U.scope other))
    U.confidence target other === Nothing
    U.finding (U.establish target other) === E.Unknown (E.TaskLossProblem mismatch)
