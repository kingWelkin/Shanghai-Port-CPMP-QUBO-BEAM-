# v5.0 longer-budget comparison and tail repair

Date: 2026-10-13

Status: COMPLETE. All frozen20/60/120-second screens, the serial-guard conditional diagnostic and independent path reviews are complete. No timed study remains active. This is a selected development/tail study, not full169 confirmation or a production default promotion.

This report will distinguish the 20/60/120-second comparisons, unchanged primary screen, explicitly selected historical tails, and any subsequently executed full169 validation. Same-budget native v4.3 will receive its complete budget. Only physically replayed, timely received complete paths count.

## 扩预算比较的当前结论

v4.3 的单次求解预算已经提高到120秒。新流程的平均路径优势仍存在；但纯Q不是中长预算下观察均值最好的配置，独立经典保护仍有价值。

同一批12个原有主组算例，每例两次新运行，平均基本移箱数如下。这12例在各预算下均两次成功，未跨预算拼接最优答案。

| 单次总预算 | v4.3 | 单轨Q | 原交替Q+Pair | Q停滞后再Pair |
|---|---:|---:|---:|---:|
|20秒|166.00|134.13|134.63|134.13|
|60秒|160.46|129.04|128.67|128.67|
|120秒|159.42|128.42|128.08|128.08|

120秒下，原Q+Pair在主组平均比v4.3少约19.65%移箱。包含已知退化例的全部27例诊断组中，v4.3为245.30，单轨Q为179.26，原Q+Pair为175.09，延后Pair为175.02。诊断组刻意增加了长尾例，不能把它的约28.6%整流程改善率外推为全hardset169的提升。

60/120秒下五种方法均54/54次可行；20秒下v4.3为54/54、新前端的四个精炼版本为52/54。增加预算后的成功率缺口消失，但**单次长尾没有清零**。120秒下仍有3次相对独立v4.3的损失，分别多4、47、91步。

Q的独立增量必须另看匹配经典对照：120秒主组12例，延后Q+Pair比Single->Pair平均少2步，区间[-3.6667,-0.5833]；全部27例平均少1.0741步，区间[-2.7963,+0.8704]跨零。不能把整流程相对v4.3的改善全部算成QUBO贡献，也不能把小诊断组当作新的全169确认。

当前新增的延后Pair调度在120秒只比原交替QP少2个“原始算例均值之和”的移箱数，尚无明显替代价值。后续修复聚焦于缺失的Single轨迹，以及独立前端给出不同起点造成的尾部差异。

## 20-second completed result

All 27 originals and 54 fresh blocks complete. All execution, hardware, physical, branch, stream and clock checks pass; 5453 complete paths are checked, with 771 producer and 752 standalone Beam commitments. Independent review replays all 262 credited answers. Native is54/54 feasible; each of the four fresh-prefix refiners is52/54. The missing prefix executions are CV10-10case15 repeat1 andcase28 repeat1. These are retained, not assigned zero moves or repaired from historical paths.

The new candidate runs cyclic Q to a full-pass stall, then pays for independent cyclic Pair from the original path. The matched classical control replaces the primary Q trajectory by Single. All budgets include both trajectories, the common producer and output cleanup.

| Cohort | Candidate versus | Common originals, both repeats | Wins/losses/ties | Sum of original-mean deltas | Worst mean loss |
|---|---|---:|---:|---:|---:|
| Unchanged primary12 | native v4.3 |12|10/0/2|-382.5|0|
| Unchanged primary12 | solo Q |12|0/0/12|0|0|
| Unchanged primary12 | original alternating Q+Pair |12|3/0/9|-6|0|
| Unchanged primary12 | deferred Single+Pair |12|7/1/4|-37|0.5|
| Prior native-relative tail9 | native v4.3 |7|5/2/0|-343.5|30|
| Prior scheduling/basin tail6 | solo Q |6|1/0/5|-10|0|
| Prior scheduling/basin tail6 | original alternating Q+Pair |6|3/1/2|-23|22|
| Entire selected screen27 | solo Q |25|1/2/22|+3.5|13|
| Entire selected screen27 | original alternating Q+Pair |25|7/2/16|-16.5|22|
| Entire selected screen27 | deferred Single+Pair |25|15/3/7|-70|21|

This is **not a passed candidate screen**. The candidate loses two feasible executions to native, is3.5 original-mean total moves worse than soloQ over common successes, and retains a repeated22-move regression to the old paired scheduler oncase32. Its matched-classical average increment remains observable: primary95%mean interval[-5.25,-1.08333], selected-screen interval[-5.1,-0.02]. These are development-cohort statistics, not a new full169 result.

The repeated physical traces explain the tradeoff:

- CV10-6case32: soloQ and deferredQ follow identical216→208→207→207 primary paths in both repeats. Deferred Pair starts at17.08/15.21 seconds and does not reach the old paired scheduler's185 before the deadline. The deferred output remains207 in both repeats. Waiting for a fully stalled primary can start the useful protection too late.
- CV10-6case34: soloQ stops at237; both paid paired variants reach227 in both repeats. The deferred variant actually starts Pair at14.65/14.58 seconds, and its winning output comes from Pair's FRG continuation. This10-move rescue is executed and paid, not projected.
- CV10-6case33: deferredQ is22 moves better than the old paired scheduler in both repeats and equals soloQ.
- CV10-6case40: deferredQ equals soloQ in both runs; one run improves19 moves over the old paired scheduler, the other ties. These are fresh outputs, not the earlier historical paths.

Of30 runs with completed overlapping solo/deferred primary cycles,29 have identical overlapping paths. One path differs under the bounded timed kernels. Two native-tail losses to soloQ occur without a Pair start; finite-time execution differences must not be misattributed to protection. Maintaining the same operator schedule does not prove per-run wall-time dominance.

The earlier native-tail selection also does not imply that every historical loss will reproduce on new time-limited paths. All comparisons above use controls executed in the same new block. The60/120-second stages continue as frozen to separate insufficient budget from trajectory/scheduling limitations.

Evidence: `screen_t20/manifest.json`, `summary.json`, `branch_audit.json`, `trajectory_identity_review.json`, `results.csv`, and the unmodified per-block `runs/` records. The20-second negative screen is retained regardless of subsequent longer-budget outcomes.

## 60-second completed result

All54 blocks complete; all five methods are54/54 feasible. The former two unavailable-prefix executions no longer create a coverage deficit in this fresh60s study. All budget/physical/hardware/branch/stream audits pass:8917 path checks,760 producer Beam checks,889 native Beam checks. Independent review replays all270 credited answers;53 of54 runs with overlapping complete solo/deferred Q cycles have identical overlapping physical paths.

| Cohort | Native v4.3 mean | SoloQ mean | Original Q+Pair mean | Deferred Q+Pair mean |
|---|---:|---:|---:|---:|
| Unchanged primary12 |160.4583|129.0417|128.6667|128.6667|
| Prior native-relative tail9 |334.2222|214.5556|220.2778|213.7778|
| Prior scheduling/basin tail6 |298.0833|220.8333|214.5000|214.5000|
| Entire selected27 |248.9630|177.9444|178.2778|176.1111|

All27 originals have both repeats feasible. The selected27 is deliberately enriched with old tails; its mean gain is not an estimate for the entire169 corpus.

DeferredQ versus soloQ is7W/0L/20T,-49.5 original-mean total moves,95%mean interval[-4.03704,-0.27778]. Versus oldQP it is4W/1L/22T,-58.5,interval[-6.14815,+0.14815];those improvements are not stable strict wins inbothrepeats. Versus matched deferredSingle it is13W/2L/12T,-53.5,interval[-4.57407,+0.77778]. The primary12 comparison to matched deferredSingle is5W/0L/7T,-37,interval[-6.66667,-0.41667]. The declared candidate expansion screen still fails; no gate is changed to call it a pass.

Against native, deferredQ is25W/0L/2T by original means. **This is not zero execution-level tail risk:** BF14case1 has deltas+1/-19 and CV10-6case33 has+26/-113. All9 old native-relative tail originals strictly improve inbothfreshrepeats, for all three Q-based variants. Therefore the removal of those repeated old tails cannot be attributed solely to the new scheduler; the longer budget is material.

On the six scheduling/basin sentinels, deferredQ equals oldQP onall6. It improves over soloQ oncase32(-23/-23),case33(-1/-1),andcase34(-14/-14). These repairs are actually executed within60s. Thus paid classical protection becomes more useful when there is time to complete it.

## Remaining classical-relative tails and next controlled diagnostic

Two original-mean losses remain against the matched classical Single->Pair control:

- CV10-10case28: candidate-relative execution deltas+26/0. On source0, the Q trajectory is still improving at58.74 seconds and the method reaches the deadline; it never gets to its Pair guard. This is a budget-limited continuation.
- CV10-6case17: deltas-8/+48. On source1, Q and Pair both finish naturally by46.83 seconds, with output242, while independently executed Single->Pair returns194. Q and Pair alone therefore do not cover the useful Single trajectory. Simply granting the same stalled trajectories more time does not change this path-dependent limitation.

The next candidate is Q->Single->Pair, always using independently paid trajectories from the original common path; Q->Pair->Single is an order control. The Q phase still runs to its own full-pass stall first. Existing Q models/kernels remain unchanged. Seven focused serial/deferred scheduling, deadline and physical tests pass (1.25 seconds).

The diagnostic was frozen to use BOTH source handoffs of both remaining loss originals, plus BF/Open family sentinels andcase32/34 Pair-rescue sentinels. It reexecutes all new refinement under matched residual allowances corresponding to60/120seconds. **Historical upstream is excluded, and the120 allowance uses the same saved60-second-produced path.** It ran after the real fresh120s suite and is not fresh end-to-end120s evidence. Completed results follow below.

## 120-second completed result

All54 blocks and270 method executions are feasible. All protocol, CPU, physical, seed, branch and deadline checks pass:9763 path checks,760 producer Beam checks and783 native Beam checks. Independent review replays all270 credited answers, andall54 overlapping completed solo/deferred Q trajectories are physically identical. The paid Pair guard starts inall54 deferred runs.

| Cohort | Native mean | SoloQ mean | OriginalQP mean | DeferredQP mean |
|---|---:|---:|---:|---:|
|Primary12|159.4167|128.4167|128.0833|128.0833|
|Prior native-tail9|301.3889|216.1667|211.6111|211.3889|
|Scheduling/basin tail6|332.9167|225.5833|214.3333|214.3333|
|Selected27|245.2963|179.2593|175.0926|175.0185|

On selected27, deferredQ versus soloQ is12W/0L/15T,-114.5 original-mean total moves,95%mean interval[-8,-1.24074]. Versus oldQP it is1W/0L/26T,-2,interval[-0.22222,0],with no stable strict original win. The predeclared expansion condition requiring repeated tail improvements over oldQP is not met. This is a useful budget result, not permission to relabel the failed schedule screen as passed.

The matched classical comparison is14W/2L/11T,-29 total,95%mean interval[-2.79630,+0.87037]. The two mean losses are CV10-10case11 (0/+29 byrepeat) andcase16 (-9/+32). These120s inputs are different fresh trajectories from the60s diagnostic inputs; they are retained and must not be replaced with later conditional repairs.

### Origin of the three native-relative execution tails

| Case / repeat | Independent v4.3 | Own v4.3 prefix, before common simplification | Actual handoff | DeferredQP output | Difference to independent native |
|---|---:|---:|---:|---:|---:|
|BF16case1 /0|115|147|143|119|+4|
|CV10-10case11 /1|293|454|436|340|+47|
|CV10-10case28 /1|341|556|549|432|+91|

All three prefix searches terminate naturally, without a118s reserve switch. The corresponding prefix/native scalar configurations are identical. The observed gap therefore exists before the new route QUBO is invoked: each new pipeline still improves its own input (24,96,117 moves saved from handoff). This does **not** establish why the independently executed Beam trajectories diverged, nor does it prove that a new restart/diversity policy would repair them. It identifies an upstream source of comparative tail risk that must be distinguished from a harmful Q refinement or missing classical guard.

Exact provenance is in `screen_t120/native_execution_tail_origin.json`. No front-end restart repair has been executed in this study. The conditional QSP/QPS diagnostic targets the separately observed Single-versus-Q/Pair trajectory limitation; it does not automatically remove these upstream tails.

## Completed serial-guard conditional diagnostic

All48 blocks and192 method executions complete at12:18:44. All are feasible; budget, hardware, physical, seed, input and serial-order audits pass. There are8219 complete-path checks and192 independent final-path replays. QSP's completed primary Q trajectory overlaps identically with deferredQP in44/48 runs. All six winning Single-guard execution rows have matching completed Q paths and matching Single paths against the independently executed controls.

Four methods are actually computed: Q->Pair (old deferred), Single->Pair (classical), Q->Single->Pair (QSP repair target), and Q->Pair->Single (QPS order control). Each trajectory starts from the given common path. All new computation and output cleanup are charged to the remaining allowance; no stored refinement answer is used.

| Given input | Residual-window label | Q->Pair | QSP | QPS | Actual result |
|---|---|---|---|---|---|
|CV10-6case17, source1|60|242 /242|194 /194|242 /209|Single-first guard repairs48 moves inboth new repeats|
|CV10-6case17, source1|120|242 /242|194 /194|194 /194|Both orders repair when enough time remains|
|CV10-10case28, source0|120|255 /255|232 /232|255 /255|Single-first guard repairs23 moves inboth new repeats|
|CV10-10case28, source0|60|258 /258|262 /258|258 /258|Repair unavailable inthis tight window;one4-move loss retained|

The60/120 labels mean nominal target minus the recorded60s prefix cost, **not** a newly executed60/120s solve from the raw bay. Results above must not replace rows in the fresh budget tables.

The additional BF/Open sentinels andboth source inputs ofPair-rescue cases32/34 preserve the old deferred result atboth allowances. All new cases/source repeats/budget repeats remain included. There is one additional2-move loss oncase17 source0 at the120 allowance (QSP181 versus deferred179). Thus the candidate is not a no-regret policy.

Across24 conditional execution rows per allowance, QSP versus deferredQP gives2W/1L/21T at60 (sum-92) and4W/1L/19T at120 (sum-140). These are repeated execution counts fromsix selected originals, not24 independent cases. Original means are reported per case in `serial_guard_diagnostic/summary.json`; no confidence claim over the full corpus is made.

The successful repairs come from the **actually executed classical Single trajectory**, after the Q primary has stalled. They demonstrate the value of preserving that alternative basin; they are not additional Q-kernel gains. Q itself remains responsible for the primary trajectory and its previously measured joint improvements. The model, route domains, seed definitions and50ms joint kernel were not changed.

## Decision and remaining work

This round answers the budget objection: fresh120s native v4.3 still has worse average paths than the Q-based pipelines on the declared27-original diagnostic, including the unchanged primary12. The longer-budget result does not by itself establish a full169 or1681 promotion.

The serial Single guard is a working **experimental repair candidate** for the demonstrated Single-basin losses, with repeated48/23-move physical repairs. It still has small2/4-move regressions and an unresolved tight-window case. Fresh broad end-to-end validation ofQSP has not been run. Existing pairedQ remains the more completely validated whole-pipeline option; no default is changed.

Three separate limitations remain visible: scarce residual time, missing alternative refinement trajectories, and different upstream Beam paths. The last can cause a+91 execution gap even though Q significantly shortens its own input. A future front-end diversity/restart comparison must charge all extra searches and strengthen the v4.3 control equally; no such result is claimed here.

Implementation: `scripts/deferred_pair_refinement.py`, `scripts/serial_classical_guards.py` and their shared-ledger adapters. Protocols: `scripts/research_v50_budget_tail.py` and `scripts/research_serial_guard_diagnostic.py`. Independent reviews: `scripts/review_v50_budget_tail.py` and `scripts/review_serial_guard_diagnostic.py`. The two focused test invocations pass13 and7 tests respectively (four overlap,16 distinct tests). Both real protocol smokes pass. All failed screens, missing20s prefixes, regressions, raw trajectories and source manifests are retained.
