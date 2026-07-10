import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function writeBlob(filePath, blob) {
  await fs.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

function addText(slide, name, text, position, style) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    name,
    position,
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = style;
  return shape;
}

async function main() {
  const summary = JSON.parse(
    await fs.readFile(path.join(__dirname, "hardness_quality_summary.json"), "utf8"),
  );
  const scatterBytes = await fs.readFile(
    path.join(__dirname, "hardness_all_metrics_grid.png"),
  );
  const coverageBytes = await fs.readFile(
    path.join(__dirname, "reference_coverage_matrix.png"),
  );

  const cv = summary.suite_summary.find((row) => row.suite === "CV");
  const open120 = summary.suite_summary.find((row) => row.suite === "Open120");
  const bf = summary.suite_summary.find((row) => row.suite === "BF");
  const byAlgorithm = (name) => summary.coverage_summary.find((row) => row.algorithm === name) || {};
  const bsFrg = byAlgorithm("BS-FRG pure");
  const wang = byAlgorithm("Wang BS-B paper");
  const classic = byAlgorithm("Classic beam");
  const astar = byAlgorithm("A* no-proof");
  const movesText = (row, key) => String(row[`${key}_moves_label`] || "moves not run").replace(/\n/g, " ");
  const formatMoves = (value) => Number(value || 0).toLocaleString("en-US", { maximumFractionDigits: 0 });
  const polishTotals = summary.polish_quality_totals || summary.totals;

  const presentation = Presentation.create({
    slideSize: { width: 1280, height: 720 },
  });
  const slide = presentation.slides.add();
  slide.background.fill = "#FFFFFF";

  slide.shapes.add({
    geometry: "rect",
    name: "top-rule",
    position: { left: 42, top: 94, width: 1196, height: 1 },
    fill: "#B8BCC4",
    line: { style: "solid", fill: "#B8BCC4", width: 0 },
  });

  addText(
    slide,
    "eyebrow",
    "CPMP V4.0 FULL SNAPSHOT | OPEN120 + BF + CV",
    { left: 42, top: 28, width: 520, height: 22 },
    { fontSize: 14, bold: true, color: "#555555" },
  );
  addText(
    slide,
    "title",
    "Hardness components expose remaining quality pressure",
    { left: 42, top: 51, width: 1120, height: 46 },
    { fontSize: 35, bold: true, color: "#000000" },
  );
  addText(
    slide,
    "subtitle",
    "V4.0 stable solves all 1,681 benchmark cases; total hardness and its components separate layout difficulty from path-length quality.",
    { left: 42, top: 111, width: 1040, height: 32 },
    { fontSize: 16, color: "#555555" },
  );

  slide.images.add({
    blob: scatterBytes,
    contentType: "image/png",
    alt: "Small-multiple scatter plots of total hardness, size, density, bad ratio, and scarcity versus relocation quality",
    fit: "contain",
    position: { left: 34, top: 154, width: 770, height: 420 },
  });

  slide.shapes.add({
    geometry: "rect",
    name: "right-rail-bg",
    position: { left: 826, top: 154, width: 412, height: 478 },
    fill: "#F2F2F2",
    line: { style: "solid", fill: "#D4D4D4", width: 1 },
  });

  addText(
    slide,
    "right-rail-title",
    "Coverage and path length",
    { left: 848, top: 176, width: 350, height: 28 },
    { fontSize: 22, bold: true, color: "#000000" },
  );

  slide.images.add({
    blob: coverageBytes,
    contentType: "image/png",
    alt: "Coverage and solved-move matrix comparing V4 stable, pure BS-FRG, Wang BS-B, classic beam, and A star references",
    fit: "contain",
    position: { left: 842, top: 207, width: 378, height: 242 },
  });

  slide.shapes.add({
    geometry: "rect",
    name: "metric-rule",
    position: { left: 848, top: 456, width: 348, height: 1 },
    fill: "#B8BCC4",
    line: { style: "solid", fill: "#B8BCC4", width: 0 },
  });

  addText(
    slide,
    "metric-1",
    `${summary.totals.solved}/${summary.totals.cases} solved`,
    { left: 848, top: 466, width: 340, height: 40 },
    { fontSize: 30, bold: true, color: "#000000" },
  );
  addText(
    slide,
    "metric-1-label",
    `${formatMoves(polishTotals.moves)} polish moves before stable guard patches`,
    { left: 848, top: 506, width: 350, height: 30 },
    { fontSize: 12.5, color: "#555555" },
  );
  addText(
    slide,
    "metric-2",
    `BS-FRG: ${bsFrg["Paper BF/CV_label"]}, ${movesText(bsFrg, "Paper BF/CV")}`,
    { left: 848, top: 540, width: 350, height: 22 },
    { fontSize: 16, bold: true, color: "#000000" },
  );
  addText(
    slide,
    "metric-2-label",
    `Real30 ${bsFrg.Real30_label}, ${movesText(bsFrg, "Real30")}; t20 reference`,
    { left: 848, top: 562, width: 350, height: 18 },
    { fontSize: 12.2, color: "#555555" },
  );
  addText(
    slide,
    "metric-3",
    `Wang BS-B: ${wang["Paper BF/CV_label"]}, ${movesText(wang, "Paper BF/CV")}`,
    { left: 848, top: 586, width: 350, height: 20 },
    { fontSize: 15.5, bold: true, color: "#FF6B35" },
  );
  addText(
    slide,
    "metric-3-label",
    `Classic ${classic["Paper BF/CV_label"]}; A* Real30 ${astar.Real30_label}, ${movesText(astar, "Real30")}`,
    { left: 848, top: 606, width: 350, height: 18 },
    { fontSize: 11.6, color: "#555555" },
  );

  addText(
    slide,
    "suite-note",
    `Open120 ${open120.solved}/${open120.cases} / BF ${bf.solved}/${bf.cases} / CV ${cv.solved}/${cv.cases}`,
    { left: 42, top: 602, width: 520, height: 24 },
    { fontSize: 15, bold: true, color: "#000000" },
  );
  addText(
    slide,
    "method-note",
    "Hardness = 0.30 size + 0.25 density + 0.30 initial bad ratio + 0.15 scarcity. V4.0 coverage uses stable all-solved rows; V4.0 path length reports polish/repaired mainline moves before guard patches.",
    { left: 42, top: 632, width: 1110, height: 42 },
    { fontSize: 13, color: "#555555" },
  );

  const preview = await presentation.export({ slide, format: "png", scale: 2 });
  await writeBlob(path.join(__dirname, "hardness_quality_one_slide.png"), preview);
  const layout = await slide.export({ format: "layout" });
  await fs.writeFile(
    path.join(__dirname, "hardness_quality_one_slide.layout.json"),
    await layout.text(),
  );
  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(
    process.env.PPTX_OUT || path.join(__dirname, "hardness_quality_one_slide.pptx"),
  );
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
