using UnityEngine;
using System.Collections.Generic;
using System.IO;
using System;
using System.Globalization;
using System.Numerics;
using UnityEngine.UIElements;
using UnityEngine.Rendering;
using System.Threading;

public class UnityToColmapExporter : MonoBehaviour
{
    [Header("Scene Root (all meshes)")]
    public GameObject sceneRoot;

    [Header("Output Settings")]
    public string outputRoot = "MyUnityScene";
    public float captureInterval = 0.2f;
    public int imageWidth = 1920;
    public int imageHeight = 1080;
    public GameObject cube;

    [Header("LiDAR Settings (optional)")]
    public int lidarNx = 32;
    public int lidarNy = 32;
    public int lidarNbins = 256;
    public float lidarMaxDistance = 20f;
    public int raysPerPixel = 32;

    [Header("Point Cloud Sampling")]
    public int pointsPerTriangle = 2;
    public float noiseStdDev = 0.005f;
    private int lastCapturedFrame = -1;

    Camera cam;
    float captureTimer = 0f;
    int frameIndex = 0;
    bool exportActivated = false;

    UnityEngine.Vector3 lastPos;
    UnityEngine.Quaternion lastRot;

    class CaptureInfo
    {
        public int id;
        public string filename;
        public UnityEngine.Quaternion q_colmap;
        public UnityEngine.Vector3 t_colmap;
        public UnityEngine.Matrix4x4 W2C;
    }

    List<CaptureInfo> captures = new List<CaptureInfo>();

    void Start()
    {
        cam = GetComponent<Camera>();
        if (cam == null)
        {
            Debug.LogError("Exporter must be attached to a camera.");
            enabled = false;
            return;
        }

        Directory.CreateDirectory(outputRoot);
        Directory.CreateDirectory(Path.Combine(outputRoot, "images"));
        Directory.CreateDirectory(Path.Combine(outputRoot, "transient"));

        lastPos = cam.transform.position;
        lastRot = cam.transform.rotation;

        Debug.Log("Initialized exporter. Waiting for camera movement…");
    }

/*
    void OnPreRender()
    {
        if (!exportActivated)
        {
            float posChange = UnityEngine.Vector3.Distance(lastPos, cam.transform.position);
            float rotChange = UnityEngine.Quaternion.Angle(lastRot, cam.transform.rotation);

            if (posChange > 0.0001f || rotChange > 0.001f)
            {
                exportActivated = true;
                Debug.Log("Camera movement detected → Export starts.");
            }

            lastPos = cam.transform.position;
            lastRot = cam.transform.rotation;
            return;
        }

        captureTimer += Time.deltaTime;
        if (captureTimer >= captureInterval)
        {
            captureTimer = 0f;
            CaptureFrame();
        }
    }
    */


    void OnEnable()
    {
        RenderPipelineManager.beginCameraRendering += OnBeginCameraRendering;
    }

    void OnDisable()
    {
        RenderPipelineManager.beginCameraRendering -= OnBeginCameraRendering;
    }

    void OnBeginCameraRendering(ScriptableRenderContext ctx, Camera camera)
    {

        if (camera.cameraType != CameraType.Game)
            return;

        if (camera != cam)
            return;

        // GUARANTEE: capture at most once per Unity frame
        if (Time.frameCount == lastCapturedFrame)
            return;

        lastCapturedFrame = Time.frameCount;

        CaptureFrame();  // safe, deterministic

    }

    private class TriRecord
    {
        public MeshFilter mf;
        public Renderer renderer;
        public UnityEngine.Vector3 v0, v1, v2;
        public UnityEngine.Vector3 n0, n1, n2;
        public Rect uvRect;
    }

    public void ExportSceneMeshOBJ(string objPath, Camera bakeCam, int atlasSize = 2048)
    {
        if (sceneRoot == null) throw new Exception("sceneRoot is null");
        if (bakeCam == null) throw new Exception("bakeCam is null");

        string dir = Path.GetDirectoryName(objPath);
        string mtlPath = Path.ChangeExtension(objPath, ".mtl");
        Directory.CreateDirectory(dir);

        // Gather meshes
        MeshFilter[] mfs = sceneRoot.GetComponentsInChildren<MeshFilter>(true);

        // --- Step 1: Build atlas packing (one rect per triangle) ---
        // We will create a global atlas. Each triangle gets a small rectangle.
        // This is COLMAP-friendly and guarantees all materials collapse into one texture.

        // Collect triangles globally
        var triRecords = new List<TriRecord>(1024);

        foreach (var mf in mfs)
        {
            var mesh = mf.sharedMesh;
            if (mesh == null) continue;
            var rend = mf.GetComponent<Renderer>();
            if (rend == null || !rend.enabled) continue;

            var tris = mesh.triangles;
            var verts = mesh.vertices;
            var norms = mesh.normals;

            // Ensure normals
            if (norms == null || norms.Length != verts.Length)
            {
                mesh.RecalculateNormals();
                norms = mesh.normals;
            }

            Transform t = mf.transform;

            // Add each triangle as a record
            for (int i = 0; i < tris.Length; i += 3)
            {
                int a = tris[i + 0];
                int b = tris[i + 1];
                int c = tris[i + 2];

                UnityEngine.Vector3 wa = t.TransformPoint(verts[a]);
                UnityEngine.Vector3 wb = t.TransformPoint(verts[b]);
                UnityEngine.Vector3 wc = t.TransformPoint(verts[c]);

                UnityEngine.Vector3 na = t.TransformDirection(norms[a]).normalized;
                UnityEngine.Vector3 nb = t.TransformDirection(norms[b]).normalized;
                UnityEngine.Vector3 nc = t.TransformDirection(norms[c]).normalized;

                triRecords.Add(new TriRecord
                {
                    mf = mf,
                    renderer = rend,
                    v0 = wa, v1 = wb, v2 = wc,
                    n0 = na, n1 = nb, n2 = nc,
                });
            }
        }

        if (triRecords.Count == 0) throw new Exception("No triangles found to export.");

        // Decide per-triangle patch resolution in atlas (tradeoff quality vs size).
        // 8x8 or 16x16 is often enough for COLMAP colors.
        const int patch = 16;
        int patchesPerRow = Mathf.Max(1, atlasSize / patch);
        int maxPatches = patchesPerRow * patchesPerRow;

        if (triRecords.Count > maxPatches)
        {
            throw new Exception(
                $"Atlas too small: need {triRecords.Count} patches but atlas can hold {maxPatches}. " +
                $"Increase atlasSize or patch size."
            );
        }

        // Create atlas texture
        Texture2D atlas = new Texture2D(atlasSize, atlasSize, TextureFormat.RGBA32, false, false);
        var atlasPixels = new Color32[atlasSize * atlasSize];
        for (int i = 0; i < atlasPixels.Length; i++) atlasPixels[i] = new Color32(0, 0, 0, 255);

        // Helper to sample the scene appearance at a world point using camera render + screen sampling
        // We’ll render once into a RenderTexture, then sample pixels by projecting points to screen.
        RenderTexture rt = new RenderTexture(atlasSize, atlasSize, 24, RenderTextureFormat.ARGB32);
        Texture2D screenGrab = new Texture2D(atlasSize, atlasSize, TextureFormat.RGBA32, false, false);

        var prevRT = RenderTexture.active;
        var prevTarget = bakeCam.targetTexture;

        bakeCam.targetTexture = rt;
        bakeCam.Render();

        RenderTexture.active = rt;
        screenGrab.ReadPixels(new Rect(0, 0, atlasSize, atlasSize), 0, 0, false);
        screenGrab.Apply(false, false);

        bakeCam.targetTexture = prevTarget;
        RenderTexture.active = prevRT;

        // Bake each triangle patch by sampling a few points on the triangle and averaging their screen colors
        for (int tid = 0; tid < triRecords.Count; tid++)
        {
            int px = (tid % patchesPerRow) * patch;
            int py = (tid / patchesPerRow) * patch;

            // Sample K points in barycentric coords
            const int K = 12;
            Color accum = Color.black;
            int valid = 0;

            for (int k = 0; k < K; k++)
            {
                float r1 = UnityEngine.Random.value;
                float r2 = UnityEngine.Random.value;
                if (r1 + r2 > 1f) { r1 = 1 - r1; r2 = 1 - r2; }

                UnityEngine.Vector3 p = triRecords[tid].v0 + r1 * (triRecords[tid].v1 - triRecords[tid].v0)
                                           + r2 * (triRecords[tid].v2 - triRecords[tid].v0);

                // Project to screen
                UnityEngine.Vector3 sp = bakeCam.WorldToScreenPoint(p);
                if (sp.z <= 0) continue;

                int sx = Mathf.RoundToInt(sp.x);
                int sy = Mathf.RoundToInt(sp.y);

                // screenGrab is same size as rt: (atlasSize x atlasSize)
                if (sx < 0 || sx >= atlasSize || sy < 0 || sy >= atlasSize) continue;

                // Sample pixel (note: ReadPixels gives bottom-left origin texture)
                Color c = screenGrab.GetPixel(sx, sy);
                accum += c;
                valid++;
            }

            Color avg = (valid > 0) ? (accum / valid) : Color.magenta; // magenta indicates "not visible"

            // Fill patch with avg color
            Color32 c32 = avg;
            for (int y = 0; y < patch; y++)
            {
                int row = (py + y) * atlasSize;
                for (int x = 0; x < patch; x++)
                {
                    atlasPixels[row + (px + x)] = c32;
                }
            }

            // Store UV rect for this triangle
            triRecords[tid].uvRect = new Rect(
                (float)px / atlasSize,
                (float)py / atlasSize,
                (float)patch / atlasSize,
                (float)patch / atlasSize
            );
        }

        // Apply atlas pixels
        atlas.SetPixels32(atlasPixels);
        atlas.Apply(false, false);

        // Write atlas PNG
        string atlasName = "baked_atlas.png";
        File.WriteAllBytes(Path.Combine(dir, atlasName), atlas.EncodeToPNG());

        // Cleanup RT
        Destroy(rt);
        Destroy(screenGrab);

        // --- Step 2: Write OBJ/MTL referencing the baked atlas ---
        using StreamWriter obj = new StreamWriter(objPath);
        using StreamWriter mtl = new StreamWriter(mtlPath);

        obj.WriteLine($"mtllib {Path.GetFileName(mtlPath)}");

        string matName = "baked_material";
        mtl.WriteLine($"newmtl {matName}");
        mtl.WriteLine("Ka 0 0 0");
        mtl.WriteLine("Ks 0 0 0");
        mtl.WriteLine("Kd 1 1 1");
        mtl.WriteLine($"map_Kd {atlasName}");
        mtl.WriteLine();

        obj.WriteLine($"usemtl {matName}");

        // OBJ needs global vertex/uv/normal lists.
        // We export triangles “unindexed” so each triangle has its own 3 verts + 3 uvs.
        int vOffset = 1;
        int vtOffset = 1;
        int vnOffset = 1;

        for (int tid = 0; tid < triRecords.Count; tid++)
        {
            var tr = triRecords[tid];

            // Write vertices
            obj.WriteLine($"v {tr.v0.x} {tr.v0.y} {tr.v0.z}");
            obj.WriteLine($"v {tr.v1.x} {tr.v1.y} {tr.v1.z}");
            obj.WriteLine($"v {tr.v2.x} {tr.v2.y} {tr.v2.z}");

            // Write normals
            obj.WriteLine($"vn {tr.n0.x} {tr.n0.y} {tr.n0.z}");
            obj.WriteLine($"vn {tr.n1.x} {tr.n1.y} {tr.n1.z}");
            obj.WriteLine($"vn {tr.n2.x} {tr.n2.y} {tr.n2.z}");

            // Write UVs: map triangle corners to corners of its atlas patch
            // Use a small inset to avoid bleeding across patches.
            const float inset = 0.08f;
            Rect r = tr.uvRect;

            UnityEngine.Vector2 uvA = new UnityEngine.Vector2(r.xMin + r.width * inset, r.yMin + r.height * inset);
            UnityEngine.Vector2 uvB = new UnityEngine.Vector2(r.xMax - r.width * inset, r.yMin + r.height * inset);
            UnityEngine.Vector2 uvC = new UnityEngine.Vector2(r.xMin + r.width * inset, r.yMax - r.height * inset);

            obj.WriteLine($"vt {uvA.x} {uvA.y}");
            obj.WriteLine($"vt {uvB.x} {uvB.y}");
            obj.WriteLine($"vt {uvC.x} {uvC.y}");

            // Write face (3 verts)
            // v/vt/vn
            obj.WriteLine(
                $"f " +
                $"{(vOffset + 0)}/{(vtOffset + 0)}/{(vnOffset + 0)} " +
                $"{(vOffset + 1)}/{(vtOffset + 1)}/{(vnOffset + 1)} " +
                $"{(vOffset + 2)}/{(vtOffset + 2)}/{(vnOffset + 2)}"
            );

            vOffset += 3;
            vtOffset += 3;
            vnOffset += 3;
        }

        Debug.Log($"Baked-atlas OBJ exported to {objPath} (atlas: {atlasName})");
    }

    Texture2D ConvertToUncompressedRGBA(Texture source)
    {
        RenderTexture rt = RenderTexture.GetTemporary(
            source.width,
            source.height,
            0,
            RenderTextureFormat.ARGB32,
            RenderTextureReadWrite.Linear
        );

        Graphics.Blit(source, rt);

        RenderTexture prev = RenderTexture.active;
        RenderTexture.active = rt;

        Texture2D tex = new Texture2D(
            source.width,
            source.height,
            TextureFormat.RGBA32,
            false
        );

        tex.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0);
        tex.Apply();

        RenderTexture.active = prev;
        RenderTexture.ReleaseTemporary(rt);

        return tex;
    }

    // ================================================================
    // CAPTURE FRAME + EXPORT COLMAP POSE
    // ================================================================
    void CaptureFrame()
    {
        string baseName = $"frame_{frameIndex:D4}";
        string pngPath = Path.Combine(outputRoot, "images", baseName + ".png");

        // ----------------------------------------------------
        // 1. Render and save RGB image (FLIP VERTICALLY for COLMAP)
        // ----------------------------------------------------
        
        RenderTexture rt = new RenderTexture(imageWidth, imageHeight, 24);
        RenderTexture prev = cam.targetTexture;
        cam.targetTexture = rt;
        cam.Render();

        RenderTexture.active = rt;
        Texture2D tex = new Texture2D(imageWidth, imageHeight, TextureFormat.RGB24, false);
        tex.ReadPixels(new Rect(0, 0, imageWidth, imageHeight), 0, 0);
        tex.Apply();

        /*

        // Flip vertically to convert Unity bottom-left origin → COLMAP top-left origin
        Color[] p = tex.GetPixels();
        Color[] p2 = new Color[p.Length];

        for (int y = 0; y < imageHeight; y++)
        {
            Array.Copy(
                p, y * imageWidth,
                p2, (imageHeight - 1 - y) * imageWidth,
                imageWidth
            );
        }
        tex.SetPixels(p2);
        tex.Apply();
        */

        File.WriteAllBytes(pngPath, tex.EncodeToPNG());

        cam.targetTexture = prev;
        RenderTexture.active = null;
        Destroy(rt);
        Destroy(tex);


        Debug.Log("fov : " + cam.fieldOfView);

        // ============================================================
        // CORRECT COLMAP POSE EXPORT
        // ============================================================

        // Unity camera-to-world matrix (left-handed)

        // FULL Unity → COLMAP conversion:
        // Flip Y and Z axes (fix handedness + fix forward axis)
        // Matrix4x4 S = Matrix4x4.Scale(new Vector3(1f, -1f, -1f));
       // Matrix4x4 C2W_colmap = S * C2W_unity;

        // Compute world→camera rotation
        // Matrix4x4 Rwc = C2W_colmap.inverse;

        //Matrix4x4 Rwc = C2W_unity.inverse;

        UnityEngine.Vector3 C = cam.transform.position;

        // Unity camera-to-world
        UnityEngine.Matrix4x4 C2W_unity = cam.cameraToWorldMatrix;

        // World flip (Unity Y-up → COLMAP Y-down)
        UnityEngine.Matrix4x4 Fy = UnityEngine.Matrix4x4.Scale(new UnityEngine.Vector3(1f, -1f, -1f));

        // Convert WORLD coordinates
        UnityEngine.Matrix4x4 C2W_colmap = Fy * C2W_unity;

        // Now invert to get WORLD → CAMERA
        UnityEngine.Matrix4x4 W2C_colmap = Fy * cam.worldToCameraMatrix;

        // Extract R (zero translation)
        UnityEngine.Matrix4x4 R_wc = W2C_colmap;
        R_wc.m03 = R_wc.m13 = R_wc.m23 = 0f;

        // Extract t directly
        UnityEngine.Vector3 t_wc = new UnityEngine.Vector3(
            W2C_colmap.m03,
            W2C_colmap.m13,
            W2C_colmap.m23
        );

        UnityEngine.Quaternion q = R_wc.rotation;

        // Recover camera center
        UnityEngine.Matrix4x4 R_cw = R_wc.transpose;
        UnityEngine.Vector3 C_recovered = -R_cw.MultiplyVector(t_wc);

        Debug.Log("Unity C:     " + C);
        Debug.Log("Recovered C: " + C_recovered);

        Debug.Log("W2C Matrix: ");
        Debug.Log(W2C_colmap.ToString());

        // Save
        captures.Add(new CaptureInfo
        {
            id = frameIndex + 1,
            filename = baseName + ".png",
            q_colmap = q,
            t_colmap = t_wc,
            W2C = W2C_colmap
        });

        SaveTransientHistogram(baseName);

        frameIndex++;
        Debug.Log($"Captured frame {baseName}");
    }


    public static UnityEngine.Matrix4x4 ColmapRotationMatrix(float qw, float qx, float qy, float qz)
    {
        // Normalize quaternion (recommended)
        float norm = Mathf.Sqrt(qw*qw + qx*qx + qy*qy + qz*qz);
        qw /= norm; qx /= norm; qy /= norm; qz /= norm;

        UnityEngine.Matrix4x4 R = new UnityEngine.Matrix4x4();

        R.m00 = 1f - 2f * (qy*qy + qz*qz);
        R.m01 = 2f * (qx*qy - qz*qw);
        R.m02 = 2f * (qx*qz + qy*qw);
        R.m03 = 0f;

        R.m10 = 2f * (qx*qy + qz*qw);
        R.m11 = 1f - 2f * (qx*qx + qz*qz);
        R.m12 = 2f * (qy*qz - qx*qw);
        R.m13 = 0f;

        R.m20 = 2f * (qx*qz - qy*qw);
        R.m21 = 2f * (qy*qz + qx*qw);
        R.m22 = 1f - 2f * (qx*qx + qy*qy);
        R.m23 = 0f;

        R.m30 = 0f;
        R.m31 = 0f;
        R.m32 = 0f;
        R.m33 = 1f;

        return R;
    }

    // Quaternion extraction from rotation matrix
    UnityEngine.Quaternion QuaternionFromMatrixCorrect(UnityEngine.Matrix4x4 m)
    {
        UnityEngine.Quaternion q = new UnityEngine.Quaternion();
        float trace = m.m00 + m.m11 + m.m22;

        if (trace > 0f)
        {
            float s = Mathf.Sqrt(trace + 1f) * 2f;
            q.w = 0.25f * s;
            q.x = (m.m21 - m.m12) / s;
            q.y = (m.m02 - m.m20) / s;
            q.z = (m.m10 - m.m01) / s;
        }
        else if (m.m00 > m.m11 && m.m00 > m.m22)
        {
            float s = Mathf.Sqrt(1f + m.m00 - m.m11 - m.m22) * 2f;
            q.w = (m.m21 - m.m12) / s;
            q.x = 0.25f * s;
            q.y = (m.m01 + m.m10) / s;
            q.z = (m.m02 + m.m20) / s;
        }
        else if (m.m11 > m.m22)
        {
            float s = Mathf.Sqrt(1f + m.m11 - m.m00 - m.m22) * 2f;
            q.w = (m.m02 - m.m20) / s;
            q.x = (m.m01 + m.m10) / s;
            q.y = 0.25f * s;
            q.z = (m.m12 + m.m21) / s;
        }
        else
        {
            float s = Mathf.Sqrt(1f + m.m22 - m.m00 - m.m11) * 2f;
            q.w = (m.m10 - m.m01) / s;
            q.x = (m.m02 + m.m20) / s;
            q.y = (m.m12 + m.m21) / s;
            q.z = 0.25f * s;
        }

        return q;
    }

    // Transient LiDAR (unchanged)
    void SaveTransientHistogram(string baseName)
    {
        float[,,] histogram = new float[lidarNx, lidarNy, lidarNbins];
        float binSize = lidarMaxDistance / lidarNbins;

        for (int ix = 0; ix < lidarNx; ix++) {
            for (int iy = 0; iy < lidarNy; iy++) {

                float u = (ix + 0.5f) / lidarNx;
                float v = (iy + 0.5f) / lidarNy;

                Ray ray = cam.ViewportPointToRay(new UnityEngine.Vector3(u, v, 0));
                if (Physics.Raycast(ray, out RaycastHit hit, lidarMaxDistance))
                {
                    float d = hit.distance;
                    int bin = Mathf.Clamp(Mathf.FloorToInt(d / binSize), 0, lidarNbins - 1);
                    histogram[ix, iy, bin] += 1f;
                }
            }
        }

        string csvPath = Path.Combine(outputRoot, "transient", baseName + "_transient.csv");
        using (StreamWriter sw = new StreamWriter(csvPath))
        {
            for (int b = 0; b < lidarNbins; b++)
            {
                string[] row = new string[lidarNx * lidarNy];
                int k = 0;

                for (int ix = 0; ix < lidarNx; ix++)
                    for (int iy = 0; iy < lidarNy; iy++)
                        row[k++] = histogram[ix, iy, b].ToString("R", CultureInfo.InvariantCulture);

                sw.WriteLine(string.Join(",", row));
            }
        }
    }

    // ================================================================
    // WRITE COLMAP FILES
    // ================================================================
    void WriteCamerasTxt()
    {
        Debug.Log("application quit");
        Debug.Log("fov : " + cam.fieldOfView);

        Debug.Log(imageHeight);
        Debug.Log(imageWidth);

        Debug.Log(cam.fieldOfView);

       // float fy = (imageHeight * 0.5f) / Mathf.Tan(cam.fieldOfView * Mathf.Deg2Rad * 0.5f);
        //float fx = fy * (imageWidth / (float)imageHeight);
        float cx = imageWidth * 0.5f;
        float cy = imageHeight * 0.5f;

        UnityEngine.Matrix4x4 P = cam.projectionMatrix;

        // focal lengths in pixel units
        float fy = P.m11 * 0.5f * cam.pixelHeight;
        float fx = P.m00 * 0.5f * cam.pixelWidth;

        Debug.Log("focal length x :");
        Debug.Log(fx);
        Debug.Log("focal length y :");
        Debug.Log(fy);

        // float f = Screen.height / (2f * Mathf.Tan(fovY / 2f));

        // Debug.Log("f is :");
        // Debug.Log(f);

        string path = Path.Combine(outputRoot, "cameras.txt");
        using StreamWriter sw = new StreamWriter(path);

        sw.WriteLine("# CAMERA_ID MODEL WIDTH HEIGHT PARAMS");
        sw.WriteLine($"1 PINHOLE {imageWidth} {imageHeight} {fx} {fy} {cx} {cy}");
    }

    void WriteImagesTxt()
    {
        string path = Path.Combine(outputRoot, "images.txt");
        using StreamWriter sw = new StreamWriter(path);

        sw.WriteLine("# IMAGE_ID W2C00 W2C01 W2C02 W2C03 W2C10 W2C11 W2C12 W2C13 W2C20 W2C21 W2C22 W2C23 W2C30 W2C31 W2C32 W2C33 CAMERA_ID NAME");

        foreach (var c in captures)
        {

            sw.WriteLine($"{c.id} {c.W2C.m00} {c.W2C.m01} {c.W2C.m02} {c.W2C.m03} {c.W2C.m10} {c.W2C.m11} {c.W2C.m12} {c.W2C.m13} " +
                                $"{c.W2C.m20} {c.W2C.m21} {c.W2C.m22} {c.W2C.m23} {c.W2C.m30} {c.W2C.m31} {c.W2C.m32} {c.W2C.m33} " +
                                $"1 {c.filename}");
            /*
            sw.WriteLine($"{c.id} {c.q_colmap.w} {c.q_colmap.x} {c.q_colmap.y} {c.q_colmap.z} " +
                         $"{c.t_colmap.x} {c.t_colmap.y} {c.t_colmap.z} 1 {c.filename}");
                         */
            sw.WriteLine();
        }
    }

    void WritePoints3DTxt()
    {
        string path = Path.Combine(outputRoot, "points3D.txt");
        using StreamWriter sw = new StreamWriter(path);

        sw.WriteLine("# POINT3D_ID X Y Z R G B ERROR");

        if (sceneRoot == null) return;

        MeshFilter[] meshes = sceneRoot.GetComponentsInChildren<MeshFilter>(true);
        int pid = 1;

        foreach (MeshFilter mf in meshes)
        {
            Mesh mesh = mf.sharedMesh;
            if (mesh == null) continue;

            UnityEngine.Vector3[] verts = mesh.vertices;
            int[] tris = mesh.triangles;

            // --- Try to get vertex colors ---
            Color[] vcolors = mesh.colors;
            bool hasVertexColors = vcolors != null && vcolors.Length == verts.Length;

            // --- Fallback: material color ---
            Color matColor = Color.white;
            Renderer rend = mf.GetComponent<Renderer>();
            if (rend != null && rend.sharedMaterial != null && rend.sharedMaterial.HasProperty("_Color"))
                matColor = rend.sharedMaterial.color;

            for (int i = 0; i < tris.Length; i += 3)
            {
                int i0 = tris[i];
                int i1 = tris[i + 1];
                int i2 = tris[i + 2];

                UnityEngine.Vector3 v0 = mf.transform.TransformPoint(verts[i0]);
                UnityEngine.Vector3 v1 = mf.transform.TransformPoint(verts[i1]);
                UnityEngine.Vector3 v2 = mf.transform.TransformPoint(verts[i2]);

                Color c0 = hasVertexColors ? vcolors[i0] : matColor;
                Color c1 = hasVertexColors ? vcolors[i1] : matColor;
                Color c2 = hasVertexColors ? vcolors[i2] : matColor;

                for (int s = 0; s < pointsPerTriangle; s++)
                {
                    float a = UnityEngine.Random.value;
                    float b = UnityEngine.Random.value;
                    if (a + b > 1f) { a = 1 - a; b = 1 - b; }

                    // --- Barycentric interpolation ---
                    UnityEngine.Vector3 p = v0 + a * (v1 - v0) + b * (v2 - v0);

                    Color c = c0 + a * (c1 - c0) + b * (c2 - c0);

                    // --- Noise ---
                    p += new UnityEngine.Vector3(
                        RandomGaussian() * noiseStdDev,
                        RandomGaussian() * noiseStdDev,
                        RandomGaussian() * noiseStdDev
                    );

                    // --- Convert color to 0–255 ---
                    int R = Mathf.Clamp(Mathf.RoundToInt(c.r * 255f), 0, 255);
                    int G = Mathf.Clamp(Mathf.RoundToInt(c.g * 255f), 0, 255);
                    int B = Mathf.Clamp(Mathf.RoundToInt(c.b * 255f), 0, 255);

                    Debug.Log("R: " + R);
                    Debug.Log("c.r: " + c.r);

                    sw.WriteLine($"{pid} {p.x} {p.y} {p.z} {R} {G} {B} 0");
                    pid++;
                }
            }
        }
    }

    float RandomGaussian()
    {
        float u1 = 1f - UnityEngine.Random.value;
        float u2 = 1f - UnityEngine.Random.value;
        return Mathf.Sqrt(-2f * Mathf.Log(u1)) * Mathf.Cos(2 * Mathf.PI * u2);
    }

    void OnApplicationQuit()
    {
        if (captures.Count == 0)
        {
            Debug.LogWarning("Nothing captured.");
            return;
        }

        WriteCamerasTxt();
        WriteImagesTxt();
        WritePoints3DTxt();
        //ExportSceneMeshOBJ(Path.Combine(outputRoot, "/Users/emreinceoglu/Desktop/scene_mesh.obj"));

        /*ExportSceneMeshOBJ(
            Path.Combine(outputRoot, "scene_mesh.obj"),
            Camera.main,
            2048
        );
        */

        Debug.Log("Export complete.");
    }
}
