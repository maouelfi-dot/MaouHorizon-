#!/usr/bin/env bash
# Jalankan di terminal Codespaces FORK-mu, di folder repo, branch "starling": bash hemat.sh
set -e
git checkout starling
git apply --whitespace=nowarn <<'PATCH'
diff --git a/crates/horizon-client/src/preparer.rs b/crates/horizon-client/src/preparer.rs
index 5fe6672..a6c94a8 100644
--- a/crates/horizon-client/src/preparer.rs
+++ b/crates/horizon-client/src/preparer.rs
@@ -187,7 +187,11 @@ fn run(resources: &Path) -> Result<(), String> {
     })
     .map_err(|e| format!("Control textures: {e}"))?;
     // Everything the game reads is in the resources now: the runtime and the work files go.
-    for done in [&jre, &kit.join(".local"), &temp] {
+    // The fetched official files (client and server jars, libraries and the whole asset store)
+    // go too: they are a second copy of what is prepared, and the setup fetches them again
+    // if the prepared resources are ever lost.
+    let fetched = crate::accounts::files_root();
+    for done in [&jre, &kit.join(".local"), &temp, &fetched] {
         let _ = std::fs::remove_dir_all(done);
     }
     Ok(())
diff --git a/crates/starling-account/src/files.rs b/crates/starling-account/src/files.rs
index 7e0b504..03a1daf 100644
--- a/crates/starling-account/src/files.rs
+++ b/crates/starling-account/src/files.rs
@@ -339,7 +339,9 @@ pub fn download(
         for object in objects["objects"]
             .as_object()
             .into_iter()
-            .flat_map(|o| o.values())
+            .flat_map(|o| o.iter())
+            .filter(|(name, _)| used_by_starling(name))
+            .map(|(_, object)| object)
         {
             let Some(hash) = object["hash"]
                 .as_str()
@@ -365,6 +367,23 @@ pub fn download(
     })
 }
 
+/// The asset names the preparation copies into the game's resources (sounds, the title
+/// panorama, fonts, the built-in packs and the languages: horizon-prepare's `sounds.rs`).
+/// The rest of the asset store (icons, texts and so on) is never read, so it is not fetched.
+fn used_by_starling(name: &str) -> bool {
+    name == "pack.mcmeta"
+        || name == "minecraft/sounds.json"
+        || [
+            "minecraft/sounds/",
+            "minecraft/textures/gui/title/background/",
+            "minecraft/font/",
+            "minecraft/resourcepacks/",
+            "minecraft/lang/",
+        ]
+        .iter()
+        .any(|prefix| name.starts_with(prefix))
+}
+
 /// Every item, a few at a time; what is already there with its hash counts as done.
 fn fetch_all(
     items: Vec<Item>,
diff --git a/packaging/android/build_apk.py b/packaging/android/build_apk.py
index 3c215d5..e3ca4fe 100644
--- a/packaging/android/build_apk.py
+++ b/packaging/android/build_apk.py
@@ -35,13 +35,13 @@ ROOT = Path(__file__).resolve().parents[2]
 HERE = Path(__file__).resolve().parent
 
 APPLICATION_ID = "io.github.juanluci.starling"
-LABEL = "Starling"
+LABEL = "Maou Horizon"
 # --dev and --preview: an app of its own, installed beside the release (which a build signed with
 # another key cannot install over), with its own folder (StarlingActivity.sharedFolder: StarlingDev,
 # StarlingPreview) and its own sign-in and game files. Preview is what starling-client/starling's
 # CI builds, signed with that repository's test key: the release key stays out of it.
 DEV_SUFFIX = ".dev"
-VARIANTS = {"dev": (DEV_SUFFIX, " Dev"), "preview": (".preview", " Preview")}
+VARIANTS = {"dev": (DEV_SUFFIX, " Dev"), "preview": (".preview", "")}
 VERSION_NAME = "0.8.0"
 VERSION_CODE = 13
 MIN_SDK = 26  # AAudio (cpal)
PATCH
git add -A
git commit -m "Hemat penyimpanan dan ganti nama jadi Maou Horizon"
git push origin starling
echo "Selesai. Buka tab Actions di fork-mu dan tunggu build 'android', lalu unduh artifact starling-apk-arm64-v8a."
