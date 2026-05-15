using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Windows.Forms;
using AxPEM3DControlLib;

internal static class Program
{
    [STAThread]
    private static void Main(string[] args)
    {
        var outDir = args.Length > 1
            ? Path.GetFullPath(args[1])
            : Path.GetFullPath(Path.Combine(".", "out"));
        Directory.CreateDirectory(outDir);

        try
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);

            var sample = args.Length > 0
                ? Path.GetFullPath(args[0])
                : Path.GetFullPath(Path.Combine("..", "samples", "1@206.ptt"));

            using var form = new ProbeForm(sample, outDir);
            Application.Run(form);
        }
        catch (Exception ex)
        {
            File.WriteAllText(
                Path.Combine(outDir, "official_ocx_probe_boot_error.json"),
                JsonSerializer.Serialize(DescribeException(ex), new JsonSerializerOptions { WriteIndented = true }));
            Console.Error.WriteLine(ex);
            Environment.ExitCode = 1;
        }
    }

    private static object DescribeException(Exception ex)
    {
        return new
        {
            type = ex.GetType().FullName,
            message = ex.Message,
            hresult = "0x" + ex.HResult.ToString("X8", CultureInfo.InvariantCulture),
            inner = ex.InnerException == null ? null : DescribeException(ex.InnerException)
        };
    }
}

internal sealed class ProbeForm : Form
{
    private readonly string _sample;
    private readonly string _outDir;
    private readonly AxPEM3DControl _control = new();

    public ProbeForm(string sample, string outDir)
    {
        _sample = sample;
        _outDir = outDir;
        Text = "Official OCX Probe";
        Width = 640;
        Height = 480;
        ShowInTaskbar = false;

        _control.Dock = DockStyle.Fill;
        Controls.Add(_control);
        Shown += (_, _) => BeginInvoke(RunProbe);
    }

    private void RunProbe()
    {
        var report = new Dictionary<string, object?>();
        try
        {
            report["sample"] = _sample;
            report["sample_exists"] = File.Exists(_sample);

            var loaded = _control.LoadFile(_sample);
            report["load_file"] = loaded;
            Application.DoEvents();

            TryCall(report, "opengl_info", () =>
            {
                string info = "";
                _control.GetOpenGLInfo(ref info);
                return info;
            });
            TryCall(report, "height_minmax", () =>
            {
                float min = 0;
                float max = 0;
                _control.GetHeightMinMax(ref min, ref max);
                return new { min, max };
            });
            TryCall(report, "height_color_range", () =>
            {
                float min = 0;
                float max = 0;
                _control.GetHeightColorRange(ref min, ref max);
                return new { min, max };
            });
            TryCall(report, "info_3d", () =>
            {
                int centerX = 0;
                int centerY = 0;
                float resolX = 0;
                float resolY = 0;
                _control.Get3DInfo(ref centerX, ref centerY, ref resolX, ref resolY);
                return new { centerX, centerY, resolX, resolY };
            });

            DumpBuffer(report, "height", () =>
            {
                int width = 0;
                int height = 0;
                short bit = 0;
                var buffer = _control.GetHeightBuffer(ref width, ref height, ref bit);
                return (buffer, width, height, bit);
            });
            DumpBuffer(report, "real", () =>
            {
                int width = 0;
                int height = 0;
                short bit = 0;
                var buffer = _control.GetRealBuffer(ref width, ref height, ref bit);
                return (buffer, width, height, bit);
            });
        }
        catch (Exception ex)
        {
            report["fatal"] = DescribeException(ex);
        }
        finally
        {
            var path = Path.Combine(_outDir, "official_ocx_probe_report.json");
            File.WriteAllText(path, JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true }));
            Close();
        }
    }

    private void DumpBuffer(Dictionary<string, object?> report, string name, Func<(object? buffer, int width, int height, short bit)> getBuffer)
    {
        TryCall(report, name + "_buffer", () =>
        {
            var (buffer, width, height, bit) = getBuffer();
            var prefix = Path.Combine(_outDir, "official_" + name);
            var meta = BufferMeta(buffer, width, height, bit, prefix);
            return meta;
        });
    }

    private static object BufferMeta(object? buffer, int width, int height, short bit, string prefix)
    {
        if (buffer is null)
        {
            return new { width, height, bit, type = "null" };
        }

        var type = buffer.GetType();
        var values = Flatten(buffer).ToArray();
        var numeric = values.Select(ToDoubleOrNull).Where(v => v.HasValue).Select(v => v!.Value).ToArray();
        File.WriteAllText(prefix + ".txt", string.Join(Environment.NewLine, values.Select(v => Convert.ToString(v, CultureInfo.InvariantCulture))));

        return new
        {
            width,
            height,
            bit,
            type = type.FullName,
            count = values.Length,
            first_values = values.Take(16).Select(v => Convert.ToString(v, CultureInfo.InvariantCulture)).ToArray(),
            numeric_min = numeric.Length == 0 ? (double?)null : numeric.Min(),
            numeric_max = numeric.Length == 0 ? (double?)null : numeric.Max(),
            numeric_mean = numeric.Length == 0 ? (double?)null : numeric.Average(),
            dump = Path.GetFullPath(prefix + ".txt")
        };
    }

    private static IEnumerable<object?> Flatten(object buffer)
    {
        if (buffer is Array array)
        {
            foreach (var item in array)
            {
                yield return item;
            }
            yield break;
        }

        if (buffer is IEnumerable enumerable && buffer is not string)
        {
            foreach (var item in enumerable)
            {
                yield return item;
            }
            yield break;
        }

        yield return buffer;
    }

    private static double? ToDoubleOrNull(object? value)
    {
        if (value is null)
        {
            return null;
        }
        try
        {
            return Convert.ToDouble(value, CultureInfo.InvariantCulture);
        }
        catch
        {
            return null;
        }
    }

    private static void TryCall(Dictionary<string, object?> report, string key, Func<object?> call)
    {
        try
        {
            report[key] = call();
        }
        catch (Exception ex)
        {
            report[key + "_error"] = DescribeException(ex);
        }
    }

    private static object DescribeException(Exception ex)
    {
        return new
        {
            type = ex.GetType().FullName,
            message = ex.Message,
            hresult = "0x" + ex.HResult.ToString("X8", CultureInfo.InvariantCulture),
            inner = ex.InnerException == null ? null : DescribeException(ex.InnerException)
        };
    }
}
