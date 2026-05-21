import 'dart:convert';
import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';

enum AnalyzeMode { combined, classification, soc }

const _defaultApiBaseUrl = 'https://agrisync-gmxy.onrender.com';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const SoilApp());
}

class SoilApp extends StatelessWidget {
  const SoilApp({super.key});

  @override
  Widget build(BuildContext context) {
    final colorScheme = ColorScheme.fromSeed(
      seedColor: const Color(0xFF2E7D5C),
      brightness: Brightness.light,
    );
    final textTheme = GoogleFonts.spaceGroteskTextTheme();
    return MaterialApp(
      title: 'Soil Organic Carbon',
      theme: ThemeData(
        colorScheme: colorScheme,
        textTheme: textTheme,
        useMaterial3: true,
        appBarTheme: const AppBarTheme(
          backgroundColor: Colors.transparent,
          elevation: 0,
          centerTitle: false,
        ),
      ),
      home: const SoilHomePage(),
    );
  }
}

class SoilHomePage extends StatefulWidget {
  const SoilHomePage({super.key});

  @override
  State<SoilHomePage> createState() => _SoilHomePageState();
}

class _SoilHomePageState extends State<SoilHomePage> {
  final _picker = ImagePicker();

  AnalyzeMode _mode = AnalyzeMode.combined;
  File? _imageFile;
  Map<String, dynamic>? _result;
  bool _isLoading = false;
  String? _error;

  @override
  void dispose() {
    super.dispose();
  }

  Future<void> _pickImage() async {
    final image = await _picker.pickImage(source: ImageSource.gallery);
    if (image == null) {
      return;
    }

    setState(() {
      _imageFile = File(image.path);
      _result = null;
      _error = null;
    });
  }

  Future<void> _openCamera() async {
    final captured = await Navigator.of(
      context,
    ).push<File?>(MaterialPageRoute(builder: (_) => const CameraCapturePage()));
    if (captured == null) {
      return;
    }

    setState(() {
      _imageFile = captured;
      _result = null;
      _error = null;
    });
  }

  String _endpointForMode(AnalyzeMode mode) {
    switch (mode) {
      case AnalyzeMode.classification:
        return '/classify';
      case AnalyzeMode.soc:
        return '/soc';
      case AnalyzeMode.combined:
        return '/analyze';
    }
  }

  Future<void> _analyze() async {
    if (_imageFile == null) {
      setState(() {
        _error = 'Pick an image first.';
      });
      return;
    }

    final baseUrl = _defaultApiBaseUrl;
    if (baseUrl.isEmpty) {
      setState(() {
        _error = 'Enter the API base URL.';
      });
      return;
    }

    setState(() {
      _isLoading = true;
      _error = null;
      _result = null;
    });

    try {
      final uri = Uri.parse('$baseUrl${_endpointForMode(_mode)}');
      final request = http.MultipartRequest('POST', uri);
      request.files.add(
        await http.MultipartFile.fromPath('file', _imageFile!.path),
      );

      final streamed = await request.send();
      final response = await http.Response.fromStream(streamed);

      if (response.statusCode >= 200 && response.statusCode < 300) {
        final data = jsonDecode(response.body) as Map<String, dynamic>;
        setState(() {
          _result = data;
        });
      } else {
        setState(() {
          _error = 'Server error: ${response.statusCode} ${response.body}';
        });
      }
    } catch (exc) {
      setState(() {
        _error = 'Request failed: $exc';
      });
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;
    return Scaffold(
      body: Stack(
        children: [
          Positioned.fill(
            child: Container(
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  begin: Alignment.topCenter,
                  end: Alignment.bottomCenter,
                  colors: [colors.surface, const Color(0xFFF2F7F4)],
                ),
              ),
            ),
          ),
          SafeArea(
            child: SingleChildScrollView(
              padding: const EdgeInsets.fromLTRB(20, 16, 20, 24),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  Text(
                    'Soil Organic Carbon',
                    style: Theme.of(context).textTheme.headlineSmall,
                  ),
                  const SizedBox(height: 4),
                  Text(
                    'Analyze soil images with classification and SOC insight.',
                    style: Theme.of(context).textTheme.bodyMedium,
                  ),
                  const SizedBox(height: 16),
                  _SectionCard(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Mode',
                          style: Theme.of(context).textTheme.titleSmall,
                        ),
                        const SizedBox(height: 12),
                        SegmentedButton<AnalyzeMode>(
                          segments: const [
                            ButtonSegment(
                              value: AnalyzeMode.combined,
                              label: Text('Combined'),
                            ),
                            ButtonSegment(
                              value: AnalyzeMode.classification,
                              label: Text('Classification'),
                            ),
                            ButtonSegment(
                              value: AnalyzeMode.soc,
                              label: Text('SOC Only'),
                            ),
                          ],
                          selected: {_mode},
                          onSelectionChanged: (selection) {
                            setState(() {
                              _mode = selection.first;
                              _result = null;
                              _error = null;
                            });
                          },
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 16),
                  _SectionCard(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Soil image',
                          style: Theme.of(context).textTheme.titleSmall,
                        ),
                        const SizedBox(height: 12),
                        if (_imageFile != null)
                          ClipRRect(
                            borderRadius: BorderRadius.circular(16),
                            child: Image.file(
                              _imageFile!,
                              height: 220,
                              width: double.infinity,
                              fit: BoxFit.cover,
                            ),
                          )
                        else
                          Container(
                            height: 180,
                            decoration: BoxDecoration(
                              color: colors.surfaceContainerHighest,
                              borderRadius: BorderRadius.circular(16),
                            ),
                            child: Column(
                              mainAxisAlignment: MainAxisAlignment.center,
                              children: [
                                Icon(
                                  Icons.image_outlined,
                                  size: 36,
                                  color: colors.onSurfaceVariant,
                                ),
                                const SizedBox(height: 8),
                                Text(
                                  'No image selected',
                                  style: Theme.of(context).textTheme.bodyMedium
                                      ?.copyWith(
                                        color: colors.onSurfaceVariant,
                                      ),
                                ),
                              ],
                            ),
                          ),
                        const SizedBox(height: 16),
                        Row(
                          children: [
                            Expanded(
                              child: FilledButton.tonalIcon(
                                onPressed: _pickImage,
                                icon: const Icon(Icons.photo_library_outlined),
                                label: const Text('Gallery'),
                              ),
                            ),
                            const SizedBox(width: 12),
                            Expanded(
                              child: FilledButton.tonalIcon(
                                onPressed: _openCamera,
                                icon: const Icon(Icons.camera_alt_outlined),
                                label: const Text('Camera'),
                              ),
                            ),
                          ],
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 16),
                  FilledButton.icon(
                    onPressed: _isLoading ? null : _analyze,
                    icon: const Icon(Icons.analytics_outlined),
                    label: Text(_isLoading ? 'Analyzing...' : 'Analyze'),
                    style: FilledButton.styleFrom(
                      padding: const EdgeInsets.symmetric(vertical: 14),
                      textStyle: const TextStyle(fontWeight: FontWeight.w600),
                    ),
                  ),
                  if (_error != null) ...[
                    const SizedBox(height: 16),
                    Text(_error!, style: TextStyle(color: colors.error)),
                  ],
                  if (_result != null) ...[
                    const SizedBox(height: 20),
                    _ResultCard(result: _result!, mode: _mode),
                  ],
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _ResultCard extends StatelessWidget {
  const _ResultCard({required this.result, required this.mode});

  final Map<String, dynamic> result;
  final AnalyzeMode mode;

  @override
  Widget build(BuildContext context) {
    final soc = result['soc'] as Map<String, dynamic>?;
    final isSoil = result['is_soil'] == true;
    final showSoc = mode != AnalyzeMode.classification;

    return _SectionCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text('Results', style: Theme.of(context).textTheme.titleSmall),
          const SizedBox(height: 12),
          Text(
            result['predicted_class'].toString(),
            style: Theme.of(context).textTheme.titleLarge,
          ),
          const SizedBox(height: 6),
          Wrap(
            spacing: 10,
            runSpacing: 10,
            children: [
              _MetricChip(
                label: 'Confidence',
                value: (result['confidence'] as num).toStringAsFixed(3),
              ),
              _MetricChip(
                label: 'Margin',
                value: (result['margin'] as num).toStringAsFixed(3),
              ),
              _MetricChip(label: 'Soil', value: isSoil ? 'Yes' : 'No'),
            ],
          ),
          if (showSoc) ...[
            const SizedBox(height: 16),
            if (soc != null) ...[
              Text(
                'SOC ${(soc['percent'] as num).toStringAsFixed(2)}%',
                style: Theme.of(context).textTheme.titleMedium,
              ),
              const SizedBox(height: 6),
              Text('g/kg ${(soc['g_per_kg'] as num).toStringAsFixed(2)}'),
              Text('Category: ${soc['category']}'),
              const SizedBox(height: 8),
              Text('${soc['note']}'),
            ] else
              Text(
                isSoil
                    ? 'SOC: model not available'
                    : 'SOC blocked: image not classified as soil',
              ),
          ],
        ],
      ),
    );
  }
}

class _MetricChip extends StatelessWidget {
  const _MetricChip({required this.label, required this.value});

  final String label;
  final String value;

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: colors.surfaceContainerHighest,
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            label,
            style: Theme.of(
              context,
            ).textTheme.labelSmall?.copyWith(color: colors.onSurfaceVariant),
          ),
          const SizedBox(height: 2),
          Text(value, style: Theme.of(context).textTheme.titleSmall),
        ],
      ),
    );
  }
}

class _SectionCard extends StatelessWidget {
  const _SectionCard({required this.child});

  final Widget child;

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: colors.surface,
        borderRadius: BorderRadius.circular(20),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.04),
            blurRadius: 18,
            offset: const Offset(0, 10),
          ),
        ],
        border: Border.all(color: colors.surfaceContainerHighest),
      ),
      child: child,
    );
  }
}

class CameraCapturePage extends StatefulWidget {
  const CameraCapturePage({super.key});

  @override
  State<CameraCapturePage> createState() => _CameraCapturePageState();
}

class _CameraCapturePageState extends State<CameraCapturePage> {
  CameraController? _controller;
  Future<void>? _initFuture;
  String? _error;
  bool _isCapturing = false;

  @override
  void initState() {
    super.initState();
    _initializeCamera();
  }

  Future<void> _initializeCamera() async {
    try {
      final cameras = await availableCameras();
      if (cameras.isEmpty) {
        setState(() {
          _error = 'No cameras available.';
        });
        return;
      }

      final selected = cameras.firstWhere(
        (camera) => camera.lensDirection == CameraLensDirection.back,
        orElse: () => cameras.first,
      );

      final controller = CameraController(
        selected,
        ResolutionPreset.high,
        enableAudio: false,
      );
      setState(() {
        _controller = controller;
        _initFuture = controller.initialize();
      });
    } catch (exc) {
      setState(() {
        _error = 'Camera error: $exc';
      });
    }
  }

  @override
  void dispose() {
    _controller?.dispose();
    super.dispose();
  }

  Future<void> _capture() async {
    final controller = _controller;
    if (controller == null || _isCapturing) {
      return;
    }

    setState(() {
      _isCapturing = true;
    });

    try {
      await _initFuture;
      final file = await controller.takePicture();
      if (!mounted) {
        return;
      }
      Navigator.of(context).pop(File(file.path));
    } catch (exc) {
      if (!mounted) {
        return;
      }
      setState(() {
        _error = 'Capture failed: $exc';
        _isCapturing = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final controller = _controller;
    return Scaffold(
      appBar: AppBar(title: const Text('Camera')),
      body: _error != null
          ? Center(child: Text(_error!))
          : controller == null
          ? const Center(child: CircularProgressIndicator())
          : FutureBuilder<void>(
              future: _initFuture,
              builder: (context, snapshot) {
                if (snapshot.connectionState != ConnectionState.done) {
                  return const Center(child: CircularProgressIndicator());
                }
                return Column(
                  children: [
                    Expanded(child: CameraPreview(controller)),
                    Padding(
                      padding: const EdgeInsets.all(16),
                      child: SizedBox(
                        width: double.infinity,
                        child: FilledButton.icon(
                          onPressed: _isCapturing ? null : _capture,
                          icon: const Icon(Icons.camera),
                          label: Text(
                            _isCapturing ? 'Capturing...' : 'Capture photo',
                          ),
                        ),
                      ),
                    ),
                  ],
                );
              },
            ),
    );
  }
}
