#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(USE_PPROF)
#include <gperftools/profiler.h>
#endif

using namespace std;

mt19937 g_engine{42};

namespace algo {
template <typename Graph>
class Kuhn {
public:
  Kuhn(const Graph& graph, int from, int to, std::vector<int>& match_to)
      : graph_(graph), match_to_(match_to), from_(from), to_(to) {}

  void Go() {
    int n = graph_.NumVertices();
    match_to_.assign(n, -1);
    colors_.assign(n, -1);

    std::vector<int> unmatched;
    unmatched.reserve(to_ - from_);
    GreedyMatch(unmatched);

    for (int color = 0; !unmatched.empty(); ++color) {
      bool ok = false;

      int j = 0;
      for (int i = 0; i < static_cast<int>(unmatched.size()); ++i) {
        const int u = unmatched[i];
        if (Match(u, color))
          ok = true;
        else
          unmatched[j++] = u;
      }
      unmatched.resize(j);

      if (!ok)
        break;
    }
  }

private:
  void GreedyMatch(std::vector<int>& unmatched) {
    for (int u = from_; u < to_; ++u) {
      bool matched = false;
      for (const auto& v : graph_.OutEdges(u)) {
        if (match_to_[v] < 0) {
          match_to_[u] = v;
          match_to_[v] = u;
          matched = true;
          break;
        }
      }
      if (!matched)
        unmatched.push_back(u);
    }
  }

  bool Match(int u, int color) {
    if (colors_[u] == color)
      return false;
    colors_[u] = color;
    for (const auto& v : graph_.OutEdges(u)) {
      const int w = match_to_[v];
      if (w < 0 || Match(w, color)) {
        match_to_[u] = v;
        match_to_[v] = u;
        return true;
      }
    }
    return false;
  }

  const Graph& graph_;
  std::vector<int>& match_to_;
  std::vector<int> colors_;
  int from_;
  int to_;
};
}  // namespace algo

enum class Ort { Horizontal, Vertical };

struct Renamer {
  int GetName(const string& s) {
    auto it = m_dict.find(s);
    if (it != m_dict.end())
      return it->second;
    const int name = m_dict.size();
    m_dict[s] = name;
    return name;
  }

  int NumNames() const { return m_dict.size(); }

  unordered_map<string, int> m_dict;
};

ostream& operator<<(ostream& os, Ort ort) {
  switch (ort) {
    case Ort::Horizontal:
      os << "Horizontal";
      break;
    case Ort::Vertical:
      os << "Vertical";
      break;
  }
  return os;
}

template <typename T>
ostream& operator<<(ostream& os, const vector<T>& vs) {
  os << "[" << vs.size() << ": ";
  for (size_t i = 0; i < vs.size(); ++i) {
    if (i != 0)
      os << ", ";
    os << vs[i];
  }
  os << "]";
  return os;
}

template <typename T>
void SortUnique(vector<T>& vs) {
  vs.erase(unique(vs.begin(), vs.end()), vs.end());
}

struct Photo {
  Photo() = default;

  template <typename Tags>
  Photo(Ort ort, Tags&& tags) : m_ort(ort), m_tags(std::forward<Tags>(tags)) {
    sort(m_tags.begin(), m_tags.end());
    m_tags.erase(unique(m_tags.begin(), m_tags.end()), m_tags.end());
  }

  bool operator<(const Photo& rhs) const {
    if (m_ort != rhs.m_ort)
      return m_ort < rhs.m_ort;
    return m_tags < rhs.m_tags;
  }

  bool operator==(const Photo& rhs) const { return m_ort == rhs.m_ort && m_tags == rhs.m_tags; }

  Ort m_ort = Ort::Horizontal;
  vector<int> m_tags;
};

ostream& operator<<(ostream& os, const Photo& photo) {
  os << "Photo [" << photo.m_ort << ", " << photo.m_tags << "]";
  return os;
}

struct Index {
  Index(const vector<Photo>& photos, int numTags) : m_index(numTags) {
    for (int i = 0; i < photos.size(); ++i) {
      for (const auto& tag : photos[i].m_tags) {
        assert(tag < m_index.size());
        m_index[tag].push_back(i);
      }
    }
  }

  int GetNumTags() const { return m_index.size(); }

  const vector<int>& GetPhotosByTag(int tag) const {
    assert(tag < m_index.size());
    return m_index[tag];
  }

  vector<vector<int>> m_index;
};

int GetScore(const vector<int>& lhs, const vector<int>& rhs) {
  int inCommon = 0;

  size_t i = 0, j = 0;
  while (i < lhs.size() && j < rhs.size()) {
    if (lhs[i] == rhs[j]) {
      ++inCommon;
      ++i;
      ++j;
    } else if (lhs[i] < rhs[j]) {
      ++i;
    } else {
      ++j;
    }
  }

  assert(lhs.size() >= inCommon);
  const int inLeft = lhs.size() - inCommon;

  assert(rhs.size() >= inCommon);
  const int inRight = rhs.size() - inCommon;

  return min(min(inLeft, inRight), inCommon);
}

int GetScore(const Photo& lhs, const Photo& rhs) { return GetScore(lhs.m_tags, rhs.m_tags); }

void PrintStats(const vector<Photo>& photos, const Index& index) {
  cerr << "Num photos: " << photos.size() << endl;
  unordered_map<int, int> tags;

  int maxTags = 0;
  int numHorizontal = 0;
  int numVertical = 0;
  for (const auto& photo : photos) {
    switch (photo.m_ort) {
      case Ort::Horizontal:
        ++numHorizontal;
        break;
      case Ort::Vertical:
        ++numVertical;
        break;
    }
    if (photo.m_tags.size() > maxTags)
      maxTags = photo.m_tags.size();
    for (const auto& tag : photo.m_tags)
      ++tags[tag];
  }
  cerr << "Num horizontal: " << numHorizontal << endl;
  cerr << "Num vertical: " << numVertical << endl;
  cerr << "Total number of tags: " << tags.size() << endl;
  cerr << "Max tags for a photo: " << maxTags << endl;

  int numUnique = 0;
  for (const auto& kv : tags) {
    if (kv.second == 1)
      ++numUnique;
  }

  cerr << "Total number of unique tags: " << numUnique << endl;

  int maxList = 0;
  double meanList = 0;
  for (int tag = 0; tag < index.GetNumTags(); ++tag) {
    const auto& list = index.GetPhotosByTag(tag);
    maxList = max<int>(maxList, list.size());
    meanList += list.size();
  }
  cerr << "Max list size: " << maxList << endl;
  cerr << "Mean list size: " << meanList / index.GetNumTags() << endl;
}

struct Graph {
  Graph(const vector<Photo>& photos, const vector<vector<int>>& index) : m_adj(2 * photos.size()) {
    for (int i = 0; i < photos.size(); ++i) {
      for (const auto& tag : photos[i].m_tags) {
        assert(tag < index.size());
        for (const auto& same : index[tag]) {
          if (same != i && GetScore(photos[i].m_tags, photos[same].m_tags) == 3)
            m_adj[i].push_back(photos.size() + same);
        }
      }
    }

    int maxSize = 0;
    int sumSize = 0;
    for (auto& adj : m_adj) {
      SortUnique(adj);
      maxSize = max<int>(maxSize, adj.size());
      sumSize += adj.size();
    }

    cerr << "MaxSize: " << maxSize << endl;
    cerr << "MeanSize: " << static_cast<double>(sumSize) / m_adj.size() << endl;
  }

  int NumVertices() const { return m_adj.size(); }

  const vector<int>& OutEdges(int u) const {
    assert(u < m_adj.size());
    return m_adj[u];
  }

  vector<vector<int>> m_adj;
};

vector<int> WithKuhn(const vector<Photo>& photos, const vector<vector<int>>& index) {
  Graph graph{photos, index};
  vector<int> matchTo;
  algo::Kuhn<Graph> kuhn(graph, 0, photos.size(), matchTo);
  kuhn.Go();

  vector<int> slides;
  vector<bool> used(photos.size());

  int numCycles = 0;
  for (int u = 0; u < photos.size(); ++u) {
    int curr = u;
    assert(curr < photos.size());
    if (!used[curr])
      ++numCycles;
    while (!used[curr]) {
      slides.push_back(curr);
      used[curr] = true;
      curr = matchTo[curr];
      if (curr < 0)
        break;
      assert(curr >= photos.size());
      curr -= photos.size();
      if (curr >= photos.size()) {
        cerr << "LOL: " << curr << endl;
      }
      assert(curr < photos.size());
    }
  }

  cerr << "Num cycles: " << numCycles << endl;

  return slides;
}

struct Slide {
  explicit Slide(int index) : m_ort(Ort::Horizontal), m_index(index) {}
  Slide(int left, int right) : m_ort(Ort::Vertical), m_left(left), m_right(right) {}

  bool operator==(const Slide& rhs) const {
    return m_ort == rhs.m_ort && m_index == rhs.m_index && m_left == rhs.m_left && m_right == rhs.m_right;
  }

  bool operator<(const Slide& rhs) const {
    if (m_ort != rhs.m_ort)
      return m_ort < rhs.m_ort;
    if (m_index != rhs.m_index)
      return m_index < rhs.m_index;
    if (m_left != rhs.m_left)
      return m_left < rhs.m_left;
    return m_right < rhs.m_right;
  }

  Ort m_ort = Ort::Horizontal;

  int m_index = 0;

  int m_left = 0;
  int m_right = 0;
};

struct Pair {
  Pair() = default;
  Pair(int left, int right, const vector<int>& elems) : m_left(left), m_right(right), m_elems(elems) {}

  int m_left = 0;
  int m_right = 0;
  vector<int> m_elems;
};

struct IndexList {
  explicit IndexList(vector<int>&& indices) : m_indices(std::move(indices)) {
    for (int i = 0; i < m_indices.size(); ++i) {
      const auto index = m_indices[i];
      assert(index >= 0);

      if (m_where.size() <= index)
        m_where.resize(index + 1, -1);
      m_where[index] = i;
    }
  }

  static IndexList FromHors(const vector<Photo>& photos) {
    vector<int> hors;
    for (int i = 0; i < photos.size(); ++i) {
      if (photos[i].m_ort == Ort::Horizontal)
        hors.push_back(i);
    }
    return IndexList{std::move(hors)};
  }

  static IndexList FromHors(const vector<Photo>& photos, const vector<bool>& used) {
    vector<int> hors;
    for (int i = 0; i < photos.size(); ++i) {
      if (!used[i] && photos[i].m_ort == Ort::Horizontal)
        hors.push_back(i);
    }
    return IndexList{std::move(hors)};
  }

  bool Empty() const { return m_indices.empty(); }

  bool Evicted(int index) const { return index >= m_where.size() || m_where[index] == -1; }

  int GetRandom() const {
    assert(!Empty());
    uniform_int_distribution<int> uid(0, m_indices.size() - 1);
    return m_indices[uid(g_engine)];
  }

  void Evict(int index) {
    assert(index < m_where.size());
    const int i = m_where[index];
    assert(i != -1);

    const int last = m_indices.back();
    m_where[last] = i;
    m_where[index] = -1;
    m_indices[i] = last;
    m_indices.pop_back();
  }

  vector<int> m_indices;
  vector<int> m_where;
};

struct Solution {
  Solution() = default;
  explicit Solution(const vector<Photo>& photos) : m_used(IndexList::FromHors(photos)) {}

  void AddHor(int index, int delta) {
    assert(!m_used.Evicted(index));
    m_prefix.emplace_back(index);
    m_used.Evict(index);
    m_score += delta;
  }

  vector<Slide> m_prefix;
  IndexList m_used;
  int m_score = 0;
};

ostream& operator<<(ostream& os, const Solution& solution) {
  os << solution.m_prefix.size() << endl;
  for (const auto& slide : solution.m_prefix) {
    switch (slide.m_ort) {
      case Ort::Horizontal:
        os << slide.m_index << endl;
        break;
      case Ort::Vertical:
        os << slide.m_left << " " << slide.m_right << endl;
        break;
    }
  }
  return os;
}

struct BeamSearch {
  static inline constexpr int kBeamSize = 100;
  static inline constexpr int kNumTrials = 10000;
  static inline constexpr int kReportDelta = 1000;

  struct Update {
    Update() = default;
    Update(int index, int score, const Solution& solution) : m_index(index), m_score(score), m_solution(&solution) {}

    bool operator<(const Update& rhs) const { return m_score > rhs.m_score; }

    int m_index = 0;
    int m_score = 0;
    const Solution* m_solution = nullptr;
  };

  BeamSearch(const vector<Photo>& photos, const Index& index) : m_photos(photos), m_index(index) {}

  Solution Solve() {
    vector<Solution> beam;
    vector<Update> updates;
    vector<Solution> nbeam;
    vector<bool> used(m_photos.size());

    {
      auto hors = IndexList::FromHors(m_photos);

      while (!hors.Empty() && beam.size() < kBeamSize) {
        const auto index = hors.GetRandom();
        hors.Evict(index);

        Solution solution{m_photos};
        solution.AddHor(index, 0 /* delta */);

        beam.push_back(std::move(solution));
      }
    }

    Solution best{m_photos};
    int lastReported = 0;

    while (!beam.empty()) {
      updates.clear();

      for (const auto& solution : beam) {
        if (solution.m_score > best.m_score) {
          best = solution;
          if (best.m_score >= lastReported + kReportDelta) {
            cerr << "Best score: " << best.m_score << endl;
            lastReported = best.m_score;
          }
        }

        auto hors = solution.m_used;
        if (hors.Empty())
          continue;

        fill(used.begin(), used.end(), false);

        assert(!solution.m_prefix.empty());
        const auto prev = solution.m_prefix.back().m_index;

        bool added = false;
        for (const auto& tag : m_photos[prev].m_tags) {
          for (const auto& next : m_index.GetPhotosByTag(tag)) {
            if (used[next])
              continue;
            if (m_photos[next].m_ort != Ort::Horizontal)
              continue;
            if (hors.Evicted(next))
              continue;
            updates.emplace_back(next, solution.m_score + GetScore(m_photos[prev], m_photos[next]), solution);
            used[next] = true;
            added = true;
          }
        }

        if (added)
          continue;

        for (int trial = 0; trial < kNumTrials && !hors.Empty(); ++trial) {
          const auto next = hors.GetRandom();
          if (used[next])
            continue;

          updates.emplace_back(next, solution.m_score + GetScore(m_photos[prev], m_photos[next]), solution);
          used[next] = true;
          hors.Evict(next);
        }
      }

      if (updates.size() > kBeamSize) {
        nth_element(updates.begin(), updates.begin() + kBeamSize, updates.end());
        updates.erase(updates.begin() + kBeamSize, updates.end());
      }

      nbeam.clear();
      for (const auto& update : updates) {
        auto solution = *update.m_solution;
        solution.AddHor(update.m_index, update.m_score - update.m_solution->m_score /* delta */);
        nbeam.push_back(std::move(solution));
      }
      beam.swap(nbeam);
    }

    return best;
  }

  const vector<Photo>& m_photos;
  const Index& m_index;
  vector<vector<int>> m_scores;
};

int main() {
#if defined(USE_PPROF)
  ProfilerStart("/tmp/Solver.prof");
#endif

  ios_base::sync_with_stdio(false);

  int numPhotos;
  cin >> numPhotos;

  Renamer renamer;
  vector<Photo> photos;
  for (int i = 0; i < numPhotos; ++i) {
    string nameOrt;
    int numTags;
    cin >> nameOrt >> numTags;

    Ort ort;
    if (nameOrt == "H") {
      ort = Ort::Horizontal;
    } else if (nameOrt == "V") {
      ort = Ort::Vertical;
    } else {
      cerr << "Unknown orientation: " << nameOrt;
      return EXIT_FAILURE;
    }

    vector<int> tags(numTags);
    string tag;
    for (int i = 0; i < numTags; ++i) {
      cin >> tag;
      tags[i] = renamer.GetName(tag);
    }

    photos.emplace_back(ort, std::move(tags));
  }

  const Index index{photos, renamer.NumNames()};

  PrintStats(photos, index);

  BeamSearch bs{photos, index};
  auto solution = bs.Solve();
  cerr << "Expected score: " << solution.m_score << endl;

  cout << solution;

// const vector<int> slides = WithKuhn(photos, index);

// cout << slides.size() << endl;

// int score = 0;
// for (int i = 0; i < slides.size(); ++i) {
//   cout << slides[i] << endl;
//   if (i != 0)
//     score += GetScore(photos[slides[i - 1]].m_tags,
//     photos[slides[i]].m_tags);
// }

// cerr << "Expected score: " << score << endl;
#if defined(USE_PPROF)
  ProfilerStop();
#endif
  return EXIT_SUCCESS;
}
