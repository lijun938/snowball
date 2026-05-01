#include "LiveControlPanel.h"

#include <Windows.h>

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cwchar>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr wchar_t kPanelClassName[] = L"GptSnowAimLiveControlPanelV2";
constexpr wchar_t kWindowTitle[] = L"Snowball v1.2.0 正式版";
constexpr int kWindowWidth = 1400;
constexpr int kWindowHeight = 900;
constexpr int kSidebarWidth = 160;
constexpr int kCardRadius = 14;
constexpr COLORREF kSidebarBg = RGB(8, 12, 8);
constexpr COLORREF kSidebarActiveAccent = RGB(0, 230, 80);
constexpr COLORREF kSidebarText = RGB(140, 155, 140);
constexpr COLORREF kSidebarActiveText = RGB(0, 255, 100);
constexpr COLORREF kContentBg = RGB(10, 14, 10);
constexpr COLORREF kCardBg = RGB(18, 24, 18);
constexpr COLORREF kCardBorder = RGB(0, 180, 60);
constexpr COLORREF kTitleColor = RGB(0, 255, 100);
constexpr COLORREF kLabelColor = RGB(200, 215, 200);
constexpr COLORREF kHelpColor = RGB(100, 130, 100);
constexpr COLORREF kSectionColor = RGB(0, 200, 70);
constexpr COLORREF kEditBg = RGB(20, 28, 20);
constexpr COLORREF kEditText = RGB(0, 255, 100);

enum TabId { kTabPreview = 0, kTabDetection, kTabAim, kTabOutput, kTabEngine, kTabDiag, kTabCount };

enum ControlId : int {
    kButtonTabBase = 2000,
    kButtonTabPreview = 2000,
    kButtonTabDetection = 2001,
    kButtonTabAim = 2002,
    kButtonTabOutput = 2003,
    kButtonTabEngine = 2004,
    kButtonTabDiag = 2005,

    kEditConf = 1001,
    kEditNms,
    kEditLabel,
    kEditCenterOffsetX,
    kEditCenterOffsetY,
    kEditVerticalBias,
    kEditDeadzone,
    kEditFovRadius,
    kEditOneEuroMinCutoff,
    kEditOneEuroBeta,
    kEditOneEuroDerivativeCutoff,
    kEditTrackConfirmFrames,
    kEditTrackLostFrames,
    kEditTrackMatchMaxCost,
    kEditTargetLockBonus,
    kEditTargetSwitchMargin,
    kEditPredictionMs,
    kEditPredictionMaxBox,
    kEditStickCurve,
    kEditResponseBoost,
    kEditStickMin,
    kEditStickMax,
    kEditPdDGain,
    kEditPdFeedForward,
    kEditPdSlew,
    kEditHoldMs,

    kCheckOneEuro = 1100,
    kCheckTracking,
    kCheckPd,
    kCheckCapturePreview,

    kButtonApplyDet = 1200,
    kButtonResetDet,
    kButtonAllLabels,
    kButtonApplyAim,
    kButtonResetAim,
    kButtonApplyOutput,
    kButtonResetOutput,
    kButtonPresetTest,
    kButtonPresetStable,
    kButtonPresetAggressive,

    kButtonStartInference = 1210,
    kButtonStopInference,
    kButtonSaveConfig,
    kButtonExportJson,

    kComboEngine = 1250,

    kStaticStatus = 1300,
    kStaticBackend,
    kStaticTarget,
    kStaticTransport,
    kStaticHint,
    kStaticLabels,
    kStaticAimStats,
    kStaticOutputStats,
    kStaticEngineStatus,
    kStaticEnginePath,
};

struct FieldDef {
    int controlId;
    const wchar_t* label;
    const wchar_t* section;
    const wchar_t* helpText;
};

constexpr FieldDef kDetectionFields[] = {
    { kEditConf, L"置信阈值", L"检测参数", L"越低检测越灵敏但误检多，建议0.35-0.50" },
    { kEditNms, L"NMS 阈值", nullptr, L"重叠框合并阈值，越低合并越积极" },
    { kEditLabel, L"锁定标签 (-1=全部)", nullptr, L"指定只跟踪某个类别，-1为不限" },
};

constexpr FieldDef kAimFields[] = {
    { kEditCenterOffsetX, L"中心偏移 X", L"基础参数 ★优先调节", L"准星X轴偏移，正值向右" },
    { kEditCenterOffsetY, L"中心偏移 Y", nullptr, L"准星Y轴偏移，正值向下" },
    { kEditVerticalBias, L"纵向瞄点偏置", nullptr, L"★ 负值瞄头部，-0.20为偏上20%" },
    { kEditDeadzone, L"死区像素", nullptr, L"★ 距离小于此值不移动，防微抖" },
    { kEditFovRadius, L"FOV 半径 (0=不限)", nullptr, L"限制瞄准范围的圆形区域半径" },
    { kEditOneEuroMinCutoff, L"1€ 最小截止频率", L"一欧元滤波(平滑)", L"越小越平滑但延迟越大" },
    { kEditOneEuroBeta, L"1€ 速度权重 (Beta)", nullptr, L"★ 越大对快速移动响应越好" },
    { kEditOneEuroDerivativeCutoff, L"1€ 导数截止频率", nullptr, L"导数信号的平滑程度" },
    { kEditTrackConfirmFrames, L"确认帧数", L"目标跟踪(锁定)", L"新目标需连续检测几帧才确认" },
    { kEditTrackLostFrames, L"丢失容忍帧数", nullptr, L"★ 目标丢失后保持跟踪多少帧" },
    { kEditTrackMatchMaxCost, L"匹配最大代价", nullptr, L"超过此值认为不是同一目标" },
    { kEditTargetLockBonus, L"锁定加分", nullptr, L"★ 越大越不容易切换目标" },
    { kEditTargetSwitchMargin, L"切换裕度", nullptr, L"★ 新目标需高出此分才会切换" },
    { kEditPredictionMs, L"预测提前量 (ms)", L"运动预测", L"预测目标未来位置的毫秒数" },
    { kEditPredictionMaxBox, L"预测最大框比例", nullptr, L"预测偏移不超过框尺寸的比例" },
};

constexpr FieldDef kOutputFields[] = {
    { kEditStickCurve, L"摇杆曲线", L"摇杆参数 ★优先调节", L"★ >1手感柔和，<1灵敏" },
    { kEditResponseBoost, L"响应增强", nullptr, L"★ 整体响应倍率，越大转越快" },
    { kEditStickMin, L"最小输出 %", nullptr, L"最小摇杆输出百分比" },
    { kEditStickMax, L"最大输出 %", nullptr, L"最大摇杆输出百分比" },
    { kEditPdDGain, L"D 增益", L"PD 控制(抗振荡)", L"微分增益，抑制过冲" },
    { kEditPdFeedForward, L"前馈", nullptr, L"前馈增益，加速响应" },
    { kEditPdSlew, L"变化率限制 (%/s)", nullptr, L"摇杆每秒最大变化量" },
    { kEditHoldMs, L"TT2 保持时间 (ms)", L"传输", L"每次指令的持续毫秒数" },
};

struct FieldLayout {
    RECT labelRect{};
    RECT editRect{};
};

std::wstring ToWide(double value, int precision) {
    std::wostringstream oss;
    oss.setf(std::ios::fixed);
    oss.precision(precision);
    oss << value;
    return oss.str();
}

std::wstring ToWideFloat(float value, int precision) {
    return ToWide(static_cast<double>(value), precision);
}

std::wstring ToWideInt(int value) {
    return std::to_wstring(value);
}

bool TryParseDouble(const std::wstring& text, double& value) {
    wchar_t* end = nullptr;
    const double parsed = std::wcstod(text.c_str(), &end);
    if (end == text.c_str() || (end != nullptr && *end != L'\0')) {
        return false;
    }
    value = parsed;
    return true;
}

bool TryParseInt(const std::wstring& text, int& value) {
    wchar_t* end = nullptr;
    const long parsed = std::wcstol(text.c_str(), &end, 10);
    if (end == text.c_str() || (end != nullptr && *end != L'\0')) {
        return false;
    }
    value = static_cast<int>(parsed);
    return true;
}

std::wstring GetWindowTextString(HWND control) {
    if (control == nullptr) {
        return {};
    }
    const int length = ::GetWindowTextLengthW(control);
    std::wstring value(static_cast<std::size_t>(length) + 1, L'\0');
    if (length > 0) {
        ::GetWindowTextW(control, value.data(), length + 1);
    }
    value.resize(static_cast<std::size_t>(length));
    return value;
}

void SetControlText(HWND parent, int controlId, const std::wstring& text) {
    if (HWND control = ::GetDlgItem(parent, controlId)) {
        ::SetWindowTextW(control, text.c_str());
    }
}

void MoveControl(HWND parent, int controlId, const RECT& rect) {
    if (HWND control = ::GetDlgItem(parent, controlId)) {
        ::MoveWindow(control, rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top, TRUE);
    }
}

int RectWidth(const RECT& rect) { return rect.right - rect.left; }
int RectHeight(const RECT& rect) { return rect.bottom - rect.top; }
RECT MakeRect(int left, int top, int right, int bottom) { return { left, top, right, bottom }; }

void ShowControl(HWND parent, int controlId, bool show) {
    if (HWND control = ::GetDlgItem(parent, controlId)) {
        ::ShowWindow(control, show ? SW_SHOW : SW_HIDE);
    }
}

void FillSolid(HDC hdc, const RECT& rect, COLORREF color) {
    HBRUSH brush = ::CreateSolidBrush(color);
    ::FillRect(hdc, &rect, brush);
    ::DeleteObject(brush);
}

void DrawRoundedRect(HDC hdc, const RECT& rect, COLORREF fillColor, COLORREF borderColor, int radius) {
    HBRUSH fillBrush = ::CreateSolidBrush(fillColor);
    HPEN borderPen = ::CreatePen(PS_SOLID, 1, borderColor);
    HGDIOBJ oldBrush = ::SelectObject(hdc, fillBrush);
    HGDIOBJ oldPen = ::SelectObject(hdc, borderPen);
    ::RoundRect(hdc, rect.left, rect.top, rect.right, rect.bottom, radius, radius);
    ::SelectObject(hdc, oldBrush);
    ::SelectObject(hdc, oldPen);
    ::DeleteObject(fillBrush);
    ::DeleteObject(borderPen);
}

void DrawTextBlock(HDC hdc, const RECT& rect, const std::wstring& text, HFONT font, COLORREF color, UINT format) {
    ::SetBkMode(hdc, TRANSPARENT);
    ::SetTextColor(hdc, color);
    HFONT oldFont = static_cast<HFONT>(::SelectObject(hdc, font));
    RECT drawRect = rect;
    ::DrawTextW(hdc, text.c_str(), -1, &drawRect, format);
    ::SelectObject(hdc, oldFont);
}

std::wstring FormatObservedLabels(const std::vector<int>& labels) {
    if (labels.empty()) {
        return L"已识别标签：暂无";
    }
    std::wostringstream oss;
    oss << L"已识别标签：";
    for (std::size_t i = 0; i < labels.size(); ++i) {
        if (i != 0) oss << L" / ";
        oss << labels[i];
    }
    return oss.str();
}

struct MetricItem {
    std::wstring title;
    std::wstring value;
    COLORREF accent = RGB(0, 230, 80);
};

std::array<MetricItem, 4> BuildMetricItems(const LiveControlPanelState& state) {
    return {
        MetricItem{ L"实时帧率", ToWide(state.fps, 1) + L" FPS", RGB(0, 200, 80) },
        MetricItem{ L"识别目标数", std::to_wstring(state.detectionCount) + L" 个", RGB(0, 180, 120) },
        MetricItem{ L"锁定状态", state.hasTarget ? L"已锁定" : L"未锁定", state.hasTarget ? RGB(0, 230, 80) : RGB(100, 130, 100) },
        MetricItem{ L"输出状态", state.outputActive ? L"正在推杆" : L"空闲", state.outputActive ? RGB(0, 200, 60) : RGB(100, 130, 100) },
    };
}

} // namespace

struct LiveControlPanel::Impl {
    HWND window = nullptr;
    HFONT titleFont = nullptr;
    HFONT headingFont = nullptr;
    HFONT bodyFont = nullptr;
    HFONT smallFont = nullptr;
    HFONT sidebarFont = nullptr;
    LiveControlPanelState stagedState;
    LiveControlPanelState latestObservedState;
    bool dirty = false;
    bool shouldClose = false;
    cv::Mat latestPreview;
    int activeTab = kTabPreview;
    std::wstring hintMessage = L"调参顺序：先调[纵向偏置][-0.20瞄头]→[死区][4-8px]→[摇杆曲线][1.0-1.3]→[响应增强][1.5-2.5]";

    static LRESULT CALLBACK WndProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam);
    void EnsureRegistered();
    bool Initialize(const LiveControlPanelState& initialState);
    void CreateFonts();
    void DestroyFonts();
    HBRUSH OnCtlColor(HDC hdc, UINT message);
    void CreateControls();
    HWND CreateEdit(int controlId);
    HWND CreateButton(int controlId, const wchar_t* text);
    HWND CreateLabel(int controlId, HFONT font);
    HWND CreateCheck(int controlId, const wchar_t* text);
    void RefreshForm();
    void UpdateTextPanels();
    void ApplyLayout();
    void SwitchTab(int tab);
    void ShowTabControls();
    bool ParseDetectionForm(LiveControlPanelState& out, std::wstring& err);
    bool ParseAimForm(LiveControlPanelState& out, std::wstring& err);
    bool ParseOutputForm(LiveControlPanelState& out, std::wstring& err);
    void ApplyPreset(int controlId);
    LRESULT OnCommand(int controlId, int notificationCode);
    void DrawSidebar(HDC hdc, const RECT& client);
    void DrawPreviewTab(HDC hdc, const RECT& content);
    void DrawFieldsTab(HDC hdc, const RECT& content, const FieldDef* fields, int count, const std::vector<FieldLayout>& layouts);
    void DrawDiagTab(HDC hdc, const RECT& content);
    void OnPaint();

    RECT ContentRect() const;
    std::vector<FieldLayout> LayoutFields(const RECT& area, const FieldDef* fields, int count, int startY);
};

LiveControlPanel::LiveControlPanel() = default;

LiveControlPanel::~LiveControlPanel() { Shutdown(); }

bool LiveControlPanel::Initialize(const LiveControlPanelState& initialState) {
    Shutdown();
    m_impl = new Impl();
    if (!m_impl->Initialize(initialState)) {
        Shutdown();
        return false;
    }
    return true;
}

void LiveControlPanel::Shutdown() {
    if (m_impl == nullptr) return;
    if (m_impl->window != nullptr) {
        ::DestroyWindow(m_impl->window);
        m_impl->window = nullptr;
    }
    m_impl->DestroyFonts();
    delete m_impl;
    m_impl = nullptr;
}

bool LiveControlPanel::PumpMessages() {
    if (m_impl == nullptr || m_impl->window == nullptr) return false;
    MSG message{};
    while (::PeekMessageW(&message, nullptr, 0, 0, PM_REMOVE)) {
        ::TranslateMessage(&message);
        ::DispatchMessageW(&message);
    }
    return !m_impl->shouldClose;
}

void LiveControlPanel::SyncObservedState(const LiveControlPanelState& state) {
    if (m_impl == nullptr) return;
    m_impl->latestObservedState = state;
    if (m_impl->window != nullptr) {
        m_impl->UpdateTextPanels();
    }
}

void LiveControlPanel::UpdatePreview(const cv::Mat& frame) {
    if (m_impl == nullptr) return;
    if (frame.empty()) {
        m_impl->latestPreview.release();
    } else {
        m_impl->latestPreview = frame.clone();
    }
    if (m_impl->window != nullptr && m_impl->activeTab == kTabPreview) {
        RECT client{};
        ::GetClientRect(m_impl->window, &client);
        RECT previewRect = { kSidebarWidth, client.top, client.right, client.bottom };
        ::InvalidateRect(m_impl->window, &previewRect, FALSE);
    }
}

bool LiveControlPanel::PullState(LiveControlPanelState& state) {
    if (m_impl == nullptr || !m_impl->dirty) return false;
    state = m_impl->stagedState;
    m_impl->dirty = false;
    return true;
}

LRESULT CALLBACK LiveControlPanel::Impl::WndProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam) {
    Impl* self = nullptr;
    if (message == WM_NCCREATE) {
        const CREATESTRUCTW* create = reinterpret_cast<const CREATESTRUCTW*>(lParam);
        self = static_cast<Impl*>(create->lpCreateParams);
        ::SetWindowLongPtrW(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(self));
        self->window = hwnd;
    } else {
        self = reinterpret_cast<Impl*>(::GetWindowLongPtrW(hwnd, GWLP_USERDATA));
    }
    if (self == nullptr) return ::DefWindowProcW(hwnd, message, wParam, lParam);

    switch (message) {
    case WM_CREATE:
        self->CreateControls();
        self->RefreshForm();
        self->ApplyLayout();
        self->ShowTabControls();
        return 0;
    case WM_COMMAND:
        return self->OnCommand(LOWORD(wParam), HIWORD(wParam));
    case WM_SIZE:
        self->ApplyLayout();
        ::InvalidateRect(hwnd, nullptr, FALSE);
        return 0;
    case WM_PAINT:
        self->OnPaint();
        return 0;
    case WM_CTLCOLORSTATIC:
    case WM_CTLCOLOREDIT:
        return reinterpret_cast<LRESULT>(self->OnCtlColor(reinterpret_cast<HDC>(wParam), message));
    case WM_ERASEBKGND:
        return 1;
    case WM_CLOSE:
        self->shouldClose = true;
        ::DestroyWindow(hwnd);
        return 0;
    case WM_DESTROY:
        self->window = nullptr;
        return 0;
    default:
        return ::DefWindowProcW(hwnd, message, wParam, lParam);
    }
}

void LiveControlPanel::Impl::EnsureRegistered() {
    static bool registered = false;
    if (registered) return;
    WNDCLASSW wc{};
    wc.lpfnWndProc = &Impl::WndProc;
    wc.hInstance = ::GetModuleHandleW(nullptr);
    wc.lpszClassName = kPanelClassName;
    wc.hCursor = ::LoadCursorW(nullptr, IDC_ARROW);
    wc.hbrBackground = reinterpret_cast<HBRUSH>(COLOR_WINDOW + 1);
    wc.style = CS_HREDRAW | CS_VREDRAW | CS_DBLCLKS;
    ::RegisterClassW(&wc);
    registered = true;
}

bool LiveControlPanel::Impl::Initialize(const LiveControlPanelState& initialState) {
    stagedState = initialState;
    latestObservedState = initialState;
    EnsureRegistered();
    CreateFonts();
    window = ::CreateWindowExW(0, kPanelClassName, kWindowTitle,
        WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN | WS_VISIBLE, 24, 20, kWindowWidth, kWindowHeight,
        nullptr, nullptr, ::GetModuleHandleW(nullptr), this);
    return window != nullptr;
}

void LiveControlPanel::Impl::CreateFonts() {
    bodyFont = ::CreateFontW(22, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE,
        DEFAULT_CHARSET, OUT_OUTLINE_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY,
        VARIABLE_PITCH, L"Microsoft YaHei UI");
    smallFont = ::CreateFontW(18, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE,
        DEFAULT_CHARSET, OUT_OUTLINE_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY,
        VARIABLE_PITCH, L"Microsoft YaHei UI");
    headingFont = ::CreateFontW(24, 0, 0, 0, FW_BOLD, FALSE, FALSE, FALSE,
        DEFAULT_CHARSET, OUT_OUTLINE_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY,
        VARIABLE_PITCH, L"Microsoft YaHei UI");
    titleFont = ::CreateFontW(32, 0, 0, 0, FW_BOLD, FALSE, FALSE, FALSE,
        DEFAULT_CHARSET, OUT_OUTLINE_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY,
        VARIABLE_PITCH, L"Microsoft YaHei UI");
    sidebarFont = ::CreateFontW(22, 0, 0, 0, FW_BOLD, FALSE, FALSE, FALSE,
        DEFAULT_CHARSET, OUT_OUTLINE_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY,
        VARIABLE_PITCH, L"Microsoft YaHei UI");
}

void LiveControlPanel::Impl::DestroyFonts() {
    for (HFONT* font : { &titleFont, &headingFont, &bodyFont, &smallFont, &sidebarFont }) {
        if (*font != nullptr) { ::DeleteObject(*font); *font = nullptr; }
    }
}

HBRUSH LiveControlPanel::Impl::OnCtlColor(HDC hdc, UINT message) {
    if (message == WM_CTLCOLOREDIT) {
        ::SetBkMode(hdc, OPAQUE);
        ::SetBkColor(hdc, kEditBg);
        ::SetTextColor(hdc, kEditText);
        static HBRUSH editBrush = ::CreateSolidBrush(kEditBg);
        return editBrush;
    }
    ::SetBkMode(hdc, TRANSPARENT);
    ::SetTextColor(hdc, kLabelColor);
    static HBRUSH contentBrush = ::CreateSolidBrush(kContentBg);
    return contentBrush;
}

HWND LiveControlPanel::Impl::CreateEdit(int controlId) {
    HWND edit = ::CreateWindowExW(WS_EX_CLIENTEDGE, L"EDIT", L"",
        WS_CHILD | ES_AUTOHSCROLL, 0, 0, 10, 10, window,
        reinterpret_cast<HMENU>(static_cast<INT_PTR>(controlId)),
        ::GetModuleHandleW(nullptr), nullptr);
    if (edit) ::SendMessageW(edit, WM_SETFONT, reinterpret_cast<WPARAM>(bodyFont), TRUE);
    return edit;
}

HWND LiveControlPanel::Impl::CreateButton(int controlId, const wchar_t* text) {
    HWND button = ::CreateWindowExW(0, L"BUTTON", text,
        WS_CHILD | BS_PUSHBUTTON, 0, 0, 10, 10, window,
        reinterpret_cast<HMENU>(static_cast<INT_PTR>(controlId)),
        ::GetModuleHandleW(nullptr), nullptr);
    if (button) ::SendMessageW(button, WM_SETFONT, reinterpret_cast<WPARAM>(bodyFont), TRUE);
    return button;
}

HWND LiveControlPanel::Impl::CreateLabel(int controlId, HFONT font) {
    HWND control = ::CreateWindowExW(0, L"STATIC", L"",
        WS_CHILD, 0, 0, 10, 10, window,
        reinterpret_cast<HMENU>(static_cast<INT_PTR>(controlId)),
        ::GetModuleHandleW(nullptr), nullptr);
    if (control && font) ::SendMessageW(control, WM_SETFONT, reinterpret_cast<WPARAM>(font), TRUE);
    return control;
}

HWND LiveControlPanel::Impl::CreateCheck(int controlId, const wchar_t* text) {
    HWND check = ::CreateWindowExW(0, L"BUTTON", text,
        WS_CHILD | BS_AUTOCHECKBOX, 0, 0, 10, 10, window,
        reinterpret_cast<HMENU>(static_cast<INT_PTR>(controlId)),
        ::GetModuleHandleW(nullptr), nullptr);
    if (check) ::SendMessageW(check, WM_SETFONT, reinterpret_cast<WPARAM>(bodyFont), TRUE);
    return check;
}

void LiveControlPanel::Impl::CreateControls() {
    const wchar_t* tabNames[] = { L"预览", L"检测", L"瞄准", L"TT2输出", L"引擎", L"诊断" };
    for (int i = 0; i < kTabCount; ++i) {
        HWND btn = ::CreateWindowExW(0, L"BUTTON", tabNames[i],
            WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON | BS_FLAT,
            0, 0, kSidebarWidth, 48, window,
            reinterpret_cast<HMENU>(static_cast<INT_PTR>(kButtonTabBase + i)),
            ::GetModuleHandleW(nullptr), nullptr);
        if (btn) ::SendMessageW(btn, WM_SETFONT, reinterpret_cast<WPARAM>(sidebarFont), TRUE);
    }

    for (const auto& f : kDetectionFields) CreateEdit(f.controlId);
    for (const auto& f : kAimFields) CreateEdit(f.controlId);
    for (const auto& f : kOutputFields) CreateEdit(f.controlId);

    CreateCheck(kCheckOneEuro, L"启用一欧元滤波");
    CreateCheck(kCheckTracking, L"启用目标跟踪");
    CreateCheck(kCheckPd, L"启用 PD 控制");
    CreateCheck(kCheckCapturePreview, L"显示采集画面");

    CreateButton(kButtonApplyDet, L"应用修改");
    CreateButton(kButtonResetDet, L"恢复默认");
    CreateButton(kButtonAllLabels, L"全部标签可锁");

    CreateButton(kButtonApplyAim, L"应用修改");
    CreateButton(kButtonResetAim, L"恢复默认");

    CreateButton(kButtonApplyOutput, L"应用修改");
    CreateButton(kButtonResetOutput, L"恢复默认");
    CreateButton(kButtonPresetTest, L"测试预设");
    CreateButton(kButtonPresetStable, L"稳妥预设");
    CreateButton(kButtonPresetAggressive, L"激进预设");

    CreateLabel(kStaticStatus, headingFont);
    CreateLabel(kStaticBackend, bodyFont);
    CreateLabel(kStaticTarget, bodyFont);
    CreateLabel(kStaticTransport, smallFont);
    CreateLabel(kStaticHint, bodyFont);
    CreateLabel(kStaticLabels, smallFont);
    CreateLabel(kStaticAimStats, smallFont);
    CreateLabel(kStaticOutputStats, smallFont);

    CreateButton(kButtonStartInference, L"启动推理");
    CreateButton(kButtonStopInference, L"停止推理");
    CreateButton(kButtonSaveConfig, L"保存参数");
    CreateButton(kButtonExportJson, L"导出JSON");
    CreateLabel(kStaticEngineStatus, headingFont);
    CreateLabel(kStaticEnginePath, bodyFont);

    {
        HWND combo = ::CreateWindowExW(WS_EX_CLIENTEDGE, L"COMBOBOX", L"",
            WS_CHILD | CBS_DROPDOWNLIST | WS_VSCROLL, 0, 0, 10, 200, window,
            reinterpret_cast<HMENU>(static_cast<INT_PTR>(kComboEngine)),
            ::GetModuleHandleW(nullptr), nullptr);
        if (combo) ::SendMessageW(combo, WM_SETFONT, reinterpret_cast<WPARAM>(bodyFont), TRUE);
    }
}

void LiveControlPanel::Impl::RefreshForm() {
    const auto& s = stagedState;
    SetControlText(window, kEditConf, ToWideFloat(s.detectionConfidenceThreshold, 3));
    SetControlText(window, kEditNms, ToWideFloat(s.detectionNmsThreshold, 3));
    SetControlText(window, kEditLabel, ToWideInt(s.labelFilter));

    SetControlText(window, kEditCenterOffsetX, ToWide(s.movement.centerOffsetX, 2));
    SetControlText(window, kEditCenterOffsetY, ToWide(s.movement.centerOffsetY, 2));
    SetControlText(window, kEditVerticalBias, ToWide(s.movement.verticalBias, 3));
    SetControlText(window, kEditDeadzone, ToWide(s.movement.deadzonePixels, 2));
    SetControlText(window, kEditFovRadius, ToWide(s.movement.fovRadius, 1));
    SetControlText(window, kEditOneEuroMinCutoff, ToWide(s.movement.aimOneEuroMinCutoff, 3));
    SetControlText(window, kEditOneEuroBeta, ToWide(s.movement.aimOneEuroBeta, 4));
    SetControlText(window, kEditOneEuroDerivativeCutoff, ToWide(s.movement.aimOneEuroDerivativeCutoff, 3));
    SetControlText(window, kEditTrackConfirmFrames, ToWideInt(s.movement.aimTrackConfirmFrames));
    SetControlText(window, kEditTrackLostFrames, ToWideInt(s.movement.aimTrackLostFrames));
    SetControlText(window, kEditTrackMatchMaxCost, ToWide(s.movement.aimTrackMatchMaxCost, 3));
    SetControlText(window, kEditTargetLockBonus, ToWide(s.movement.aimTargetLockBonus, 3));
    SetControlText(window, kEditTargetSwitchMargin, ToWide(s.movement.aimTargetSwitchMargin, 3));
    SetControlText(window, kEditPredictionMs, ToWide(s.movement.aimPredictionMs, 1));
    SetControlText(window, kEditPredictionMaxBox, ToWide(s.movement.aimPredictionMaxBoxFraction, 3));

    SetControlText(window, kEditStickCurve, ToWide(s.movement.titanTwoStickCurve, 3));
    SetControlText(window, kEditResponseBoost, ToWide(s.movement.titanTwoStickResponseBoost, 3));
    SetControlText(window, kEditStickMin, ToWide(s.movement.titanTwoStickMinPercent, 2));
    SetControlText(window, kEditStickMax, ToWide(s.movement.titanTwoStickMaxPercent, 2));
    SetControlText(window, kEditPdDGain, ToWide(s.movement.titanTwoStickDerivativeGain, 4));
    SetControlText(window, kEditPdFeedForward, ToWide(s.movement.titanTwoStickFeedForward, 4));
    SetControlText(window, kEditPdSlew, ToWide(s.movement.titanTwoStickSlewPercentPerSecond, 1));
    SetControlText(window, kEditHoldMs, ToWideInt(s.movement.titanTwoHoldMs));

    if (HWND h = ::GetDlgItem(window, kCheckOneEuro))
        ::SendMessageW(h, BM_SETCHECK, s.movement.aimOneEuroEnabled ? BST_CHECKED : BST_UNCHECKED, 0);
    if (HWND h = ::GetDlgItem(window, kCheckTracking))
        ::SendMessageW(h, BM_SETCHECK, s.movement.aimTrackingEnabled ? BST_CHECKED : BST_UNCHECKED, 0);
    if (HWND h = ::GetDlgItem(window, kCheckPd))
        ::SendMessageW(h, BM_SETCHECK, s.movement.titanTwoStickPdEnabled ? BST_CHECKED : BST_UNCHECKED, 0);
    if (HWND h = ::GetDlgItem(window, kCheckCapturePreview))
        ::SendMessageW(h, BM_SETCHECK, s.capturePreviewEnabled ? BST_CHECKED : BST_UNCHECKED, 0);

    if (HWND combo = ::GetDlgItem(window, kComboEngine)) {
        ::SendMessageW(combo, CB_RESETCONTENT, 0, 0);
        int selectedIndex = 0;
        for (std::size_t i = 0; i < s.availableEngines.size(); ++i) {
            std::wstring wide(s.availableEngines[i].begin(), s.availableEngines[i].end());
            ::SendMessageW(combo, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(wide.c_str()));
            if (s.availableEngines[i] == s.selectedEnginePath) {
                selectedIndex = static_cast<int>(i);
            }
        }
        if (!s.availableEngines.empty()) {
            ::SendMessageW(combo, CB_SETCURSEL, static_cast<WPARAM>(selectedIndex), 0);
        }
    }

    UpdateTextPanels();
}

void LiveControlPanel::Impl::UpdateTextPanels() {
    if (window == nullptr) return;
    const auto& st = latestObservedState;

    std::wostringstream status;
    status << L"状态：" << (st.transportReady ? L"链路正常" : L"链路等待")
        << L" | " << (st.hasTarget ? L"已锁定目标" : L"未锁定")
        << L" | " << (st.outputActive ? L"输出中" : L"空闲");
    SetControlText(window, kStaticStatus, status.str());

    std::wstring backend = st.backendName.empty() ? L"未知"
        : std::wstring(st.backendName.begin(), st.backendName.end());
    SetControlText(window, kStaticBackend, L"链路：天创 QCAP -> " + backend + L" -> TT2");

    std::wostringstream target;
    target << L"目标标签：" << (st.currentTargetLabel >= 0 ? std::to_wstring(st.currentTargetLabel) : L"无")
        << L"    目标点：(" << st.targetX << L", " << st.targetY << L")"
        << L"    原点：(" << st.originX << L", " << st.originY << L")";
    SetControlText(window, kStaticTarget, target.str());

    std::wostringstream transport;
    transport << L"TT2 发包：dispatch " << st.transportDispatchCount
        << L" / skip " << st.transportSkipCount
        << L" / submit " << ToWide(st.transportLastSubmitMs, 2) << L" ms";
    SetControlText(window, kStaticTransport, transport.str());

    SetControlText(window, kStaticHint, hintMessage);
    SetControlText(window, kStaticLabels, FormatObservedLabels(st.observedLabels));

    std::wostringstream aim;
    aim << L"偏移：raw(" << ToWide(st.rawDeltaX, 1) << L", " << ToWide(st.rawDeltaY, 1)
        << L")  delta(" << st.deltaX << L", " << st.deltaY << L")  score " << ToWideFloat(st.score, 2);
    SetControlText(window, kStaticAimStats, aim.str());

    std::wostringstream output;
    output << L"摇杆：X " << ToWide(st.stickXPercent, 1) << L"% / Y " << ToWide(st.stickYPercent, 1) << L"%";
    SetControlText(window, kStaticOutputStats, output.str());

    SetControlText(window, kStaticEngineStatus,
        st.inferenceRunning ? L"推理状态：运行中" : L"推理状态：已停止");
    std::wstring enginePathW = st.selectedEnginePath.empty() ? L"未选择"
        : std::wstring(st.selectedEnginePath.begin(), st.selectedEnginePath.end());
    SetControlText(window, kStaticEnginePath, L"当前引擎：" + enginePathW);
}

RECT LiveControlPanel::Impl::ContentRect() const {
    RECT client{};
    if (window) ::GetClientRect(window, &client);
    return MakeRect(kSidebarWidth, client.top, client.right, client.bottom);
}

std::vector<FieldLayout> LiveControlPanel::Impl::LayoutFields(
    const RECT& area, const FieldDef* fields, int count, int startY) {
    std::vector<FieldLayout> layouts;
    layouts.reserve(count);
    int y = startY;
    const int left = area.left + 24;
    const int right = area.right - 24;
    for (int i = 0; i < count; ++i) {
        if (fields[i].section != nullptr) {
            y += 22;
        }
        RECT labelRect = MakeRect(left, y + 4, right - 200, y + 32);
        RECT editRect = MakeRect(right - 180, y, right, y + 34);
        layouts.push_back({ labelRect, editRect });
        MoveControl(window, fields[i].controlId, editRect);
        y += 44;
    }
    return layouts;
}

void LiveControlPanel::Impl::ApplyLayout() {
    if (window == nullptr) return;
    RECT client{};
    ::GetClientRect(window, &client);

    for (int i = 0; i < kTabCount; ++i) {
        RECT tabRect = MakeRect(0, 60 + i * 52, kSidebarWidth, 60 + i * 52 + 48);
        MoveControl(window, kButtonTabBase + i, tabRect);
    }

    RECT content = MakeRect(kSidebarWidth + 20, client.top + 16, client.right - 16, client.bottom - 16);

    if (activeTab == kTabPreview) {
        MoveControl(window, kCheckCapturePreview, MakeRect(content.left + 24, content.top + 8, content.left + 240, content.top + 34));
    } else if (activeTab == kTabDetection) {
        auto layouts = LayoutFields(content, kDetectionFields,
            static_cast<int>(sizeof(kDetectionFields) / sizeof(kDetectionFields[0])), content.top + 50);
        int btnY = content.top + 50 + static_cast<int>(layouts.size()) * 36 + 30;
        int bw = 140;
        MoveControl(window, kButtonApplyDet, MakeRect(content.left + 24, btnY, content.left + 24 + bw, btnY + 38));
        MoveControl(window, kButtonResetDet, MakeRect(content.left + 24 + bw + 12, btnY, content.left + 24 + bw * 2 + 12, btnY + 38));
        MoveControl(window, kButtonAllLabels, MakeRect(content.left + 24 + (bw + 12) * 2, btnY, content.left + 24 + bw * 3 + 24, btnY + 38));
        MoveControl(window, kStaticLabels, MakeRect(content.left + 24, btnY + 50, content.right - 24, btnY + 78));
    } else if (activeTab == kTabAim) {
        int startY = content.top + 50;
        MoveControl(window, kCheckOneEuro, MakeRect(content.left + 24, startY, content.left + 240, startY + 26));
        MoveControl(window, kCheckTracking, MakeRect(content.left + 260, startY, content.left + 480, startY + 26));
        startY += 34;
        auto layouts = LayoutFields(content, kAimFields,
            static_cast<int>(sizeof(kAimFields) / sizeof(kAimFields[0])), startY);
        int btnY = startY + static_cast<int>(layouts.size()) * 36 + 20;
        int bw = 140;
        MoveControl(window, kButtonApplyAim, MakeRect(content.left + 24, btnY, content.left + 24 + bw, btnY + 38));
        MoveControl(window, kButtonResetAim, MakeRect(content.left + 24 + bw + 12, btnY, content.left + 24 + bw * 2 + 12, btnY + 38));
    } else if (activeTab == kTabOutput) {
        int startY = content.top + 50;
        MoveControl(window, kCheckPd, MakeRect(content.left + 24, startY, content.left + 260, startY + 26));
        startY += 34;
        auto layouts = LayoutFields(content, kOutputFields,
            static_cast<int>(sizeof(kOutputFields) / sizeof(kOutputFields[0])), startY);
        int btnY = startY + static_cast<int>(layouts.size()) * 36 + 20;
        int bw = 140;
        MoveControl(window, kButtonApplyOutput, MakeRect(content.left + 24, btnY, content.left + 24 + bw, btnY + 38));
        MoveControl(window, kButtonResetOutput, MakeRect(content.left + 24 + bw + 12, btnY, content.left + 24 + bw * 2 + 12, btnY + 38));
        btnY += 52;
        MoveControl(window, kButtonPresetTest, MakeRect(content.left + 24, btnY, content.left + 24 + bw, btnY + 38));
        MoveControl(window, kButtonPresetStable, MakeRect(content.left + 24 + bw + 12, btnY, content.left + 24 + bw * 2 + 12, btnY + 38));
        MoveControl(window, kButtonPresetAggressive, MakeRect(content.left + 24 + (bw + 12) * 2, btnY, content.left + 24 + bw * 3 + 24, btnY + 38));
    } else if (activeTab == kTabEngine) {
        int y = content.top + 50;
        int bw = 140;
        MoveControl(window, kStaticEngineStatus, MakeRect(content.left + 24, y, content.right - 24, y + 32));
        y += 40;
        MoveControl(window, kStaticEnginePath, MakeRect(content.left + 24, y, content.right - 24, y + 28));
        y += 40;
        MoveControl(window, kComboEngine, MakeRect(content.left + 24, y, content.left + 500, y + 32));
        y += 46;
        MoveControl(window, kButtonStartInference, MakeRect(content.left + 24, y, content.left + 24 + bw, y + 38));
        MoveControl(window, kButtonStopInference, MakeRect(content.left + 24 + bw + 12, y, content.left + 24 + bw * 2 + 12, y + 38));
        y += 60;
        MoveControl(window, kButtonSaveConfig, MakeRect(content.left + 24, y, content.left + 24 + bw, y + 38));
        MoveControl(window, kButtonExportJson, MakeRect(content.left + 24 + bw + 12, y, content.left + 24 + bw * 2 + 12, y + 38));
    } else if (activeTab == kTabDiag) {
        RECT area = MakeRect(content.left + 24, content.top + 50, content.right - 24, content.bottom);
        int y = area.top;
        int rowH = 32;
        MoveControl(window, kStaticStatus, MakeRect(area.left, y, area.right, y + rowH)); y += rowH;
        MoveControl(window, kStaticBackend, MakeRect(area.left, y, area.right, y + rowH)); y += rowH;
        MoveControl(window, kStaticTarget, MakeRect(area.left, y, area.right, y + rowH)); y += rowH;
        MoveControl(window, kStaticTransport, MakeRect(area.left, y, area.right, y + rowH)); y += rowH;
        MoveControl(window, kStaticHint, MakeRect(area.left, y, area.right, y + rowH)); y += rowH;
        MoveControl(window, kStaticLabels, MakeRect(area.left, y, area.right, y + rowH)); y += rowH;
        MoveControl(window, kStaticAimStats, MakeRect(area.left, y, area.right, y + rowH)); y += rowH;
        MoveControl(window, kStaticOutputStats, MakeRect(area.left, y, area.right, y + rowH));
    }
}

void LiveControlPanel::Impl::SwitchTab(int tab) {
    if (tab == activeTab) return;
    activeTab = tab;
    ShowTabControls();
    ApplyLayout();
    if (window) ::InvalidateRect(window, nullptr, FALSE);
}

void LiveControlPanel::Impl::ShowTabControls() {
    auto showSet = [&](const FieldDef* fields, int count, bool show) {
        for (int i = 0; i < count; ++i) ShowControl(window, fields[i].controlId, show);
    };

    bool det = activeTab == kTabDetection;
    bool aim = activeTab == kTabAim;
    bool out = activeTab == kTabOutput;
    bool eng = activeTab == kTabEngine;
    bool diag = activeTab == kTabDiag;

    showSet(kDetectionFields, static_cast<int>(sizeof(kDetectionFields) / sizeof(kDetectionFields[0])), det);
    ShowControl(window, kButtonApplyDet, det);
    ShowControl(window, kButtonResetDet, det);
    ShowControl(window, kButtonAllLabels, det);

    showSet(kAimFields, static_cast<int>(sizeof(kAimFields) / sizeof(kAimFields[0])), aim);
    ShowControl(window, kCheckOneEuro, aim);
    ShowControl(window, kCheckTracking, aim);
    ShowControl(window, kButtonApplyAim, aim);
    ShowControl(window, kButtonResetAim, aim);

    showSet(kOutputFields, static_cast<int>(sizeof(kOutputFields) / sizeof(kOutputFields[0])), out);
    ShowControl(window, kCheckPd, out);
    ShowControl(window, kButtonApplyOutput, out);
    ShowControl(window, kButtonResetOutput, out);
    ShowControl(window, kButtonPresetTest, out);
    ShowControl(window, kButtonPresetStable, out);
    ShowControl(window, kButtonPresetAggressive, out);

    ShowControl(window, kButtonStartInference, eng);
    ShowControl(window, kButtonStopInference, eng);
    ShowControl(window, kComboEngine, eng);
    ShowControl(window, kStaticEngineStatus, eng);
    ShowControl(window, kStaticEnginePath, eng);
    ShowControl(window, kButtonSaveConfig, eng);
    ShowControl(window, kButtonExportJson, eng);

    ShowControl(window, kStaticLabels, det || diag);
    ShowControl(window, kStaticStatus, diag);
    ShowControl(window, kStaticBackend, diag);
    ShowControl(window, kStaticTarget, diag);
    ShowControl(window, kStaticTransport, diag);
    ShowControl(window, kStaticHint, diag);
    ShowControl(window, kStaticAimStats, diag);
    ShowControl(window, kStaticOutputStats, diag);
    ShowControl(window, kCheckCapturePreview, activeTab == kTabPreview);
}

bool LiveControlPanel::Impl::ParseDetectionForm(LiveControlPanelState& out, std::wstring& err) {
    out = stagedState;
    auto pd = [&](int id, const wchar_t* label, float& v) -> bool {
        std::wstring t = GetWindowTextString(::GetDlgItem(window, id));
        double d = 0; if (!TryParseDouble(t, d)) { err = std::wstring(label) + L" 不是有效数字"; return false; }
        v = static_cast<float>(d); return true;
    };
    if (!pd(kEditConf, L"置信阈值", out.detectionConfidenceThreshold)) return false;
    if (!pd(kEditNms, L"NMS", out.detectionNmsThreshold)) return false;
    std::wstring lt = GetWindowTextString(::GetDlgItem(window, kEditLabel));
    int lv = -1; if (!TryParseInt(lt, lv)) { err = L"锁定标签不是有效整数"; return false; }
    out.labelFilter = lv;
    out.detectionConfidenceThreshold = std::clamp(out.detectionConfidenceThreshold, 0.01f, 0.95f);
    out.detectionNmsThreshold = std::clamp(out.detectionNmsThreshold, 0.05f, 0.95f);
    return true;
}

bool LiveControlPanel::Impl::ParseAimForm(LiveControlPanelState& out, std::wstring& err) {
    out = stagedState;
    auto pd = [&](int id, const wchar_t* label, double& v) -> bool {
        std::wstring t = GetWindowTextString(::GetDlgItem(window, id));
        if (!TryParseDouble(t, v)) { err = std::wstring(label) + L" 不是有效数字"; return false; }
        return true;
    };
    auto pi = [&](int id, const wchar_t* label, int& v) -> bool {
        std::wstring t = GetWindowTextString(::GetDlgItem(window, id));
        if (!TryParseInt(t, v)) { err = std::wstring(label) + L" 不是有效整数"; return false; }
        return true;
    };
    if (!pd(kEditCenterOffsetX, L"偏移X", out.movement.centerOffsetX)) return false;
    if (!pd(kEditCenterOffsetY, L"偏移Y", out.movement.centerOffsetY)) return false;
    if (!pd(kEditVerticalBias, L"纵向偏置", out.movement.verticalBias)) return false;
    if (!pd(kEditDeadzone, L"死区", out.movement.deadzonePixels)) return false;
    if (!pd(kEditFovRadius, L"FOV", out.movement.fovRadius)) return false;
    if (!pd(kEditOneEuroMinCutoff, L"1€截止", out.movement.aimOneEuroMinCutoff)) return false;
    if (!pd(kEditOneEuroBeta, L"1€Beta", out.movement.aimOneEuroBeta)) return false;
    if (!pd(kEditOneEuroDerivativeCutoff, L"1€导数截止", out.movement.aimOneEuroDerivativeCutoff)) return false;
    if (!pi(kEditTrackConfirmFrames, L"确认帧", out.movement.aimTrackConfirmFrames)) return false;
    if (!pi(kEditTrackLostFrames, L"丢失帧", out.movement.aimTrackLostFrames)) return false;
    if (!pd(kEditTrackMatchMaxCost, L"匹配代价", out.movement.aimTrackMatchMaxCost)) return false;
    if (!pd(kEditTargetLockBonus, L"锁定加分", out.movement.aimTargetLockBonus)) return false;
    if (!pd(kEditTargetSwitchMargin, L"切换裕度", out.movement.aimTargetSwitchMargin)) return false;
    if (!pd(kEditPredictionMs, L"预测ms", out.movement.aimPredictionMs)) return false;
    if (!pd(kEditPredictionMaxBox, L"预测框比例", out.movement.aimPredictionMaxBoxFraction)) return false;

    out.movement.aimOneEuroEnabled = (::SendMessageW(::GetDlgItem(window, kCheckOneEuro), BM_GETCHECK, 0, 0) == BST_CHECKED);
    out.movement.aimTrackingEnabled = (::SendMessageW(::GetDlgItem(window, kCheckTracking), BM_GETCHECK, 0, 0) == BST_CHECKED);
    return true;
}

bool LiveControlPanel::Impl::ParseOutputForm(LiveControlPanelState& out, std::wstring& err) {
    out = stagedState;
    auto pd = [&](int id, const wchar_t* label, double& v) -> bool {
        std::wstring t = GetWindowTextString(::GetDlgItem(window, id));
        if (!TryParseDouble(t, v)) { err = std::wstring(label) + L" 不是有效数字"; return false; }
        return true;
    };
    auto pi = [&](int id, const wchar_t* label, int& v) -> bool {
        std::wstring t = GetWindowTextString(::GetDlgItem(window, id));
        if (!TryParseInt(t, v)) { err = std::wstring(label) + L" 不是有效整数"; return false; }
        return true;
    };
    if (!pd(kEditStickCurve, L"曲线", out.movement.titanTwoStickCurve)) return false;
    if (!pd(kEditResponseBoost, L"响应增强", out.movement.titanTwoStickResponseBoost)) return false;
    if (!pd(kEditStickMin, L"最小%", out.movement.titanTwoStickMinPercent)) return false;
    if (!pd(kEditStickMax, L"最大%", out.movement.titanTwoStickMaxPercent)) return false;
    if (!pd(kEditPdDGain, L"D增益", out.movement.titanTwoStickDerivativeGain)) return false;
    if (!pd(kEditPdFeedForward, L"前馈", out.movement.titanTwoStickFeedForward)) return false;
    if (!pd(kEditPdSlew, L"变化率", out.movement.titanTwoStickSlewPercentPerSecond)) return false;
    if (!pi(kEditHoldMs, L"保持ms", out.movement.titanTwoHoldMs)) return false;

    out.movement.titanTwoStickPdEnabled = (::SendMessageW(::GetDlgItem(window, kCheckPd), BM_GETCHECK, 0, 0) == BST_CHECKED);
    return true;
}

void LiveControlPanel::Impl::ApplyPreset(int controlId) {
    if (controlId == kButtonPresetTest) {
        stagedState.detectionConfidenceThreshold = 0.40f;
        stagedState.detectionNmsThreshold = 0.45f;
        stagedState.labelFilter = -1;
        stagedState.movement.deadzonePixels = 4.0;
        stagedState.movement.verticalBias = -0.10;
        stagedState.movement.titanTwoStickCurve = 0.95;
        stagedState.movement.titanTwoStickResponseBoost = 2.30;
        stagedState.movement.titanTwoStickMinPercent = 4.5;
        stagedState.movement.titanTwoStickMaxPercent = 100.0;
        hintMessage = L"已切到测试预设。";
    } else if (controlId == kButtonPresetStable) {
        stagedState.detectionConfidenceThreshold = 0.48f;
        stagedState.detectionNmsThreshold = 0.42f;
        stagedState.labelFilter = -1;
        stagedState.movement.deadzonePixels = 7.0;
        stagedState.movement.verticalBias = -0.18;
        stagedState.movement.titanTwoStickCurve = 1.20;
        stagedState.movement.titanTwoStickResponseBoost = 1.80;
        stagedState.movement.titanTwoStickMinPercent = 6.5;
        stagedState.movement.titanTwoStickMaxPercent = 82.0;
        hintMessage = L"已切到稳妥预设。";
    } else if (controlId == kButtonPresetAggressive) {
        stagedState.detectionConfidenceThreshold = 0.35f;
        stagedState.detectionNmsThreshold = 0.50f;
        stagedState.labelFilter = -1;
        stagedState.movement.deadzonePixels = 2.0;
        stagedState.movement.verticalBias = -0.22;
        stagedState.movement.titanTwoStickCurve = 0.78;
        stagedState.movement.titanTwoStickResponseBoost = 2.80;
        stagedState.movement.titanTwoStickMinPercent = 8.0;
        stagedState.movement.titanTwoStickMaxPercent = 100.0;
        hintMessage = L"已切到激进预设。";
    }
    dirty = true;
    RefreshForm();
}

LRESULT LiveControlPanel::Impl::OnCommand(int controlId, int notificationCode) {
    if (controlId >= kButtonTabBase && controlId < kButtonTabBase + kTabCount) {
        SwitchTab(controlId - kButtonTabBase);
        return 0;
    }
    if (notificationCode != BN_CLICKED) return 0;

    if (controlId == kCheckCapturePreview) {
        stagedState.capturePreviewEnabled = (::SendMessageW(::GetDlgItem(window, kCheckCapturePreview), BM_GETCHECK, 0, 0) == BST_CHECKED);
        dirty = true;
        return 0;
    }

    if (controlId == kButtonPresetTest || controlId == kButtonPresetStable || controlId == kButtonPresetAggressive) {
        ApplyPreset(controlId);
        return 0;
    }

    if (controlId == kButtonApplyDet) {
        LiveControlPanelState parsed;
        std::wstring err;
        if (!ParseDetectionForm(parsed, err)) { hintMessage = L"输入有误：" + err; UpdateTextPanels(); return 0; }
        stagedState = parsed;
        stagedState.defaults = latestObservedState.defaults;
        stagedState.defaultDetectionConfidenceThreshold = latestObservedState.defaultDetectionConfidenceThreshold;
        stagedState.defaultDetectionNmsThreshold = latestObservedState.defaultDetectionNmsThreshold;
        hintMessage = L"检测参数已应用。";
        dirty = true;
        UpdateTextPanels();
        return 0;
    }
    if (controlId == kButtonApplyAim) {
        LiveControlPanelState parsed;
        std::wstring err;
        if (!ParseAimForm(parsed, err)) { hintMessage = L"输入有误：" + err; UpdateTextPanels(); return 0; }
        stagedState = parsed;
        hintMessage = L"瞄准参数已应用。";
        dirty = true;
        UpdateTextPanels();
        return 0;
    }
    if (controlId == kButtonApplyOutput) {
        LiveControlPanelState parsed;
        std::wstring err;
        if (!ParseOutputForm(parsed, err)) { hintMessage = L"输入有误：" + err; UpdateTextPanels(); return 0; }
        stagedState = parsed;
        hintMessage = L"输出参数已应用。";
        dirty = true;
        UpdateTextPanels();
        return 0;
    }

    if (controlId == kButtonResetDet || controlId == kButtonResetAim || controlId == kButtonResetOutput) {
        stagedState.movement = latestObservedState.defaults;
        stagedState.defaults = latestObservedState.defaults;
        stagedState.detectionConfidenceThreshold = latestObservedState.defaultDetectionConfidenceThreshold;
        stagedState.detectionNmsThreshold = latestObservedState.defaultDetectionNmsThreshold;
        stagedState.defaultDetectionConfidenceThreshold = latestObservedState.defaultDetectionConfidenceThreshold;
        stagedState.defaultDetectionNmsThreshold = latestObservedState.defaultDetectionNmsThreshold;
        stagedState.labelFilter = -1;
        hintMessage = L"已恢复默认。";
        dirty = true;
        RefreshForm();
        return 0;
    }

    if (controlId == kButtonAllLabels) {
        stagedState.labelFilter = -1;
        SetControlText(window, kEditLabel, L"-1");
        hintMessage = L"已切到全部标签可锁。";
        dirty = true;
        UpdateTextPanels();
        return 0;
    }

    if (controlId == kButtonStartInference) {
        stagedState.engineAction = PanelEngineAction::StartInference;
        if (HWND combo = ::GetDlgItem(window, kComboEngine)) {
            int sel = static_cast<int>(::SendMessageW(combo, CB_GETCURSEL, 0, 0));
            if (sel >= 0 && sel < static_cast<int>(stagedState.availableEngines.size())) {
                stagedState.selectedEnginePath = stagedState.availableEngines[sel];
            }
        }
        hintMessage = L"正在启动推理引擎...";
        dirty = true;
        UpdateTextPanels();
        return 0;
    }
    if (controlId == kButtonStopInference) {
        stagedState.engineAction = PanelEngineAction::StopInference;
        hintMessage = L"正在停止推理引擎...";
        dirty = true;
        UpdateTextPanels();
        return 0;
    }
    if (controlId == kButtonSaveConfig) {
        stagedState.engineAction = PanelEngineAction::SaveConfig;
        hintMessage = L"正在保存参数到 config.ini...";
        dirty = true;
        UpdateTextPanels();
        return 0;
    }
    if (controlId == kButtonExportJson) {
        stagedState.engineAction = PanelEngineAction::ExportJson;
        hintMessage = L"正在导出参数为 JSON...";
        dirty = true;
        UpdateTextPanels();
        return 0;
    }

    return 0;
}

void LiveControlPanel::Impl::DrawSidebar(HDC hdc, const RECT& client) {
    RECT sidebar = MakeRect(client.left, client.top, kSidebarWidth, client.bottom);
    FillSolid(hdc, sidebar, kSidebarBg);

    RECT titleRect = MakeRect(12, 14, kSidebarWidth - 4, 50);
    DrawTextBlock(hdc, titleRect, L"AI", sidebarFont, RGB(0, 255, 100), DT_LEFT | DT_SINGLELINE | DT_VCENTER);

    for (int i = 0; i < kTabCount; ++i) {
        RECT tabRect = MakeRect(0, 60 + i * 52, kSidebarWidth, 60 + i * 52 + 48);
        if (i == activeTab) {
            FillSolid(hdc, MakeRect(tabRect.left, tabRect.top, tabRect.left + 4, tabRect.bottom), kSidebarActiveAccent);
            FillSolid(hdc, MakeRect(tabRect.left + 4, tabRect.top, tabRect.right, tabRect.bottom), RGB(16, 28, 16));
        }
    }
}

void LiveControlPanel::Impl::DrawPreviewTab(HDC hdc, const RECT& content) {
    RECT titleRect = MakeRect(content.left + 20, content.top + 8, content.right - 20, content.top + 38);
    DrawTextBlock(hdc, titleRect, L"实时预览", headingFont, kTitleColor, DT_LEFT | DT_SINGLELINE | DT_VCENTER);

    RECT previewArea = MakeRect(content.left + 16, content.top + 44, content.right - 16, content.bottom - 140);
    DrawRoundedRect(hdc, previewArea, RGB(14, 18, 14), RGB(0, 120, 40), 12);

    if (!latestObservedState.capturePreviewEnabled) {
        DrawTextBlock(hdc, previewArea, L"采集画面已关闭", headingFont, RGB(100, 130, 100),
            DT_CENTER | DT_VCENTER | DT_SINGLELINE);
    } else if (latestPreview.empty()) {
        DrawTextBlock(hdc, previewArea, L"等待实时画面...", headingFont, RGB(0, 200, 70),
            DT_CENTER | DT_VCENTER | DT_SINGLELINE);
    } else {
        cv::Mat dibFrame;
        if (latestPreview.channels() == 3) dibFrame = latestPreview;
        else if (latestPreview.channels() == 4) cv::cvtColor(latestPreview, dibFrame, cv::COLOR_BGRA2BGR);
        else cv::cvtColor(latestPreview, dibFrame, cv::COLOR_GRAY2BGR);

        BITMAPINFO bmi{};
        bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
        bmi.bmiHeader.biWidth = dibFrame.cols;
        bmi.bmiHeader.biHeight = -dibFrame.rows;
        bmi.bmiHeader.biPlanes = 1;
        bmi.bmiHeader.biBitCount = 24;
        bmi.bmiHeader.biCompression = BI_RGB;

        int tw = RectWidth(previewArea) - 16;
        int th = RectHeight(previewArea) - 16;
        double scale = (std::min)(static_cast<double>(tw) / dibFrame.cols, static_cast<double>(th) / dibFrame.rows);
        int dw = (std::max)(1, static_cast<int>(dibFrame.cols * scale));
        int dh = (std::max)(1, static_cast<int>(dibFrame.rows * scale));
        int dx = previewArea.left + (RectWidth(previewArea) - dw) / 2;
        int dy = previewArea.top + (RectHeight(previewArea) - dh) / 2;

        ::SetStretchBltMode(hdc, HALFTONE);
        ::StretchDIBits(hdc, dx, dy, dw, dh, 0, 0, dibFrame.cols, dibFrame.rows,
            dibFrame.data, &bmi, DIB_RGB_COLORS, SRCCOPY);
    }

    RECT metricsArea = MakeRect(content.left + 16, content.bottom - 148, content.right - 16, content.bottom - 38);
    auto metrics = BuildMetricItems(latestObservedState);
    int gap = 12;
    int cardW = (RectWidth(metricsArea) - gap * 3) / 4;
    for (int i = 0; i < 4; ++i) {
        RECT card = MakeRect(metricsArea.left + i * (cardW + gap), metricsArea.top,
            metricsArea.left + i * (cardW + gap) + cardW, metricsArea.bottom);
        DrawRoundedRect(hdc, card, RGB(18, 24, 18), RGB(0, 140, 50), 12);
        RECT accentBar = MakeRect(card.left, card.top, card.left + 5, card.bottom);
        FillSolid(hdc, accentBar, metrics[i].accent);

        RECT tRect = MakeRect(card.left + 16, card.top + 14, card.right - 8, card.top + 38);
        DrawTextBlock(hdc, tRect, metrics[i].title, smallFont, kHelpColor, DT_LEFT | DT_SINGLELINE | DT_VCENTER);
        RECT vRect = MakeRect(card.left + 16, card.top + 44, card.right - 8, card.bottom - 8);
        DrawTextBlock(hdc, vRect, metrics[i].value, headingFont, kTitleColor, DT_LEFT | DT_SINGLELINE | DT_VCENTER);
    }

    std::wstring backend = latestObservedState.backendName.empty() ? L"--"
        : std::wstring(latestObservedState.backendName.begin(), latestObservedState.backendName.end());
    std::wostringstream statusBar;
    statusBar << L"画面采集: QCAP    推理模式: " << backend
        << L"    模型尺寸: 640×640    接收帧率: " << ToWide(latestObservedState.fps, 1) << L" FPS";
    RECT statusRect = MakeRect(content.left + 16, content.bottom - 30, content.right - 16, content.bottom - 8);
    DrawTextBlock(hdc, statusRect, statusBar.str(), smallFont, kHelpColor, DT_LEFT | DT_SINGLELINE | DT_VCENTER);
}

void LiveControlPanel::Impl::DrawFieldsTab(HDC hdc, const RECT& content,
    const FieldDef* fields, int count, const std::vector<FieldLayout>& layouts) {
    const wchar_t* currentSection = nullptr;
    for (int i = 0; i < count; ++i) {
        if (fields[i].section != nullptr && fields[i].section != currentSection) {
            currentSection = fields[i].section;
            RECT sectionRect = layouts[i].labelRect;
            sectionRect.top -= 16;
            sectionRect.bottom = sectionRect.top + 22;
            DrawTextBlock(hdc, sectionRect, currentSection, bodyFont, kSectionColor, DT_LEFT | DT_SINGLELINE | DT_VCENTER);
        }
        DrawTextBlock(hdc, layouts[i].labelRect, fields[i].label, bodyFont, kLabelColor,
            DT_LEFT | DT_SINGLELINE | DT_VCENTER);
        if (fields[i].helpText != nullptr) {
            RECT helpRect = layouts[i].labelRect;
            helpRect.top = layouts[i].labelRect.bottom - 2;
            helpRect.bottom = helpRect.top + 16;
            DrawTextBlock(hdc, helpRect, fields[i].helpText, smallFont, kHelpColor,
                DT_LEFT | DT_SINGLELINE | DT_VCENTER);
        }
    }
}

void LiveControlPanel::Impl::DrawDiagTab(HDC hdc, const RECT& content) {
    RECT titleRect = MakeRect(content.left + 20, content.top + 8, content.right - 20, content.top + 38);
    DrawTextBlock(hdc, titleRect, L"运行诊断", headingFont, kTitleColor, DT_LEFT | DT_SINGLELINE | DT_VCENTER);

    RECT modelHintRect = MakeRect(content.left + 24, content.bottom - 52, content.right - 24, content.bottom - 16);
    DrawTextBlock(hdc, modelHintRect, L"模型和采集设备在 config.ini 中配置，修改后重启程序生效。", smallFont, kHelpColor,
        DT_LEFT | DT_SINGLELINE | DT_VCENTER);
}

void LiveControlPanel::Impl::OnPaint() {
    PAINTSTRUCT ps{};
    HDC hdc = ::BeginPaint(window, &ps);
    RECT client{};
    ::GetClientRect(window, &client);
    int width = RectWidth(client);
    int height = RectHeight(client);

    auto drawScene = [&](HDC dc) {
        FillSolid(dc, client, kContentBg);
        DrawSidebar(dc, client);

        RECT content = MakeRect(kSidebarWidth + 20, client.top + 16, client.right - 16, client.bottom - 16);

        if (activeTab == kTabPreview) {
            DrawPreviewTab(dc, content);
        } else if (activeTab == kTabDetection) {
            RECT titleRect = MakeRect(content.left + 20, content.top + 8, content.right - 20, content.top + 38);
            DrawTextBlock(dc, titleRect, L"检测参数", headingFont, kTitleColor, DT_LEFT | DT_SINGLELINE | DT_VCENTER);
            auto layouts = LayoutFields(content, kDetectionFields,
                static_cast<int>(sizeof(kDetectionFields) / sizeof(kDetectionFields[0])), content.top + 50);
            DrawFieldsTab(dc, content, kDetectionFields,
                static_cast<int>(sizeof(kDetectionFields) / sizeof(kDetectionFields[0])), layouts);
        } else if (activeTab == kTabAim) {
            RECT titleRect = MakeRect(content.left + 20, content.top + 8, content.right - 20, content.top + 38);
            DrawTextBlock(dc, titleRect, L"瞄准参数", headingFont, kTitleColor, DT_LEFT | DT_SINGLELINE | DT_VCENTER);
            int startY = content.top + 50 + 34;
            auto layouts = LayoutFields(content, kAimFields,
                static_cast<int>(sizeof(kAimFields) / sizeof(kAimFields[0])), startY);
            DrawFieldsTab(dc, content, kAimFields,
                static_cast<int>(sizeof(kAimFields) / sizeof(kAimFields[0])), layouts);
        } else if (activeTab == kTabOutput) {
            RECT titleRect = MakeRect(content.left + 20, content.top + 8, content.right - 20, content.top + 38);
            DrawTextBlock(dc, titleRect, L"TT2 输出", headingFont, kTitleColor, DT_LEFT | DT_SINGLELINE | DT_VCENTER);
            int startY = content.top + 50 + 34;
            auto layouts = LayoutFields(content, kOutputFields,
                static_cast<int>(sizeof(kOutputFields) / sizeof(kOutputFields[0])), startY);
            DrawFieldsTab(dc, content, kOutputFields,
                static_cast<int>(sizeof(kOutputFields) / sizeof(kOutputFields[0])), layouts);
        } else if (activeTab == kTabEngine) {
            RECT titleRect = MakeRect(content.left + 20, content.top + 8, content.right - 20, content.top + 38);
            DrawTextBlock(dc, titleRect, L"推理引擎 / 参数管理", headingFont, kTitleColor, DT_LEFT | DT_SINGLELINE | DT_VCENTER);
        } else if (activeTab == kTabDiag) {
            DrawDiagTab(dc, content);
        }
    };

    HDC bufferDc = nullptr;
    HBITMAP bufferBitmap = nullptr;
    HGDIOBJ oldBitmap = nullptr;
    if (width > 0 && height > 0) {
        bufferDc = ::CreateCompatibleDC(hdc);
        bufferBitmap = ::CreateCompatibleBitmap(hdc, width, height);
        if (bufferDc && bufferBitmap) {
            oldBitmap = ::SelectObject(bufferDc, bufferBitmap);
            drawScene(bufferDc);
            ::BitBlt(hdc, 0, 0, width, height, bufferDc, 0, 0, SRCCOPY);
            ::SelectObject(bufferDc, oldBitmap);
        } else {
            drawScene(hdc);
        }
    } else {
        drawScene(hdc);
    }
    if (bufferBitmap) ::DeleteObject(bufferBitmap);
    if (bufferDc) ::DeleteDC(bufferDc);
    ::EndPaint(window, &ps);
}
